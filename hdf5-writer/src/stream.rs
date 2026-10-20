//! Streaming emission: write the data of large datasets chunk by chunk (or slice by slice)
//! instead of holding the whole file in memory.
//!
//! [`Hdf5StreamWriter::create`] lays out the superblock and every object header first —
//! their sizes do not depend on the data — and leaves them as placeholders. Datasets marked
//! [`DatasetBuilder::streamed`] then receive their data through [`Hdf5StreamWriter::write_chunk`]
//! or [`Hdf5StreamWriter::write_slice`]: each chunk is filtered and appended as soon as it is
//! complete, so peak memory is the chunks in progress. [`Hdf5StreamWriter::finish`] appends the
//! datasets that carried their data in the plan, the global heap and every chunk index, then
//! rewrites the headers and the superblock with the final addresses (the headers keep their
//! placeholder lengths, which is checked).
//!
//! Chunked streamed datasets may grow along unlimited dimensions
//! ([`Hdf5StreamWriter::set_extent`]); chunks never written read back as the fill value.
//! Contiguous streamed datasets are pre-filled with the fill value when first written.

use super::*;
use std::collections::BTreeMap;

pub(crate) fn validate_streamed(dataset: &DatasetBuilder) -> Result<()> {
    if matches!(dataset.layout, PlannedLayout::Compact) {
        return Err(Error::UnsupportedFeature(format!(
            "compact dataset '{}' cannot be streamed",
            dataset.name
        )));
    }
    if matches!(
        dataset.datatype,
        Datatype::VarLen { .. }
            | Datatype::String {
                size: StringSize::Variable,
                ..
            }
    ) {
        return Err(Error::UnsupportedFeature(format!(
            "variable-length dataset '{}' cannot be streamed",
            dataset.name
        )));
    }
    Ok(())
}

/// Addresses of everything placed before the data.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HeaderLayout {
    root_address: u64,
    root_len: usize,
    group_addresses: Vec<u64>,
    group_lens: Vec<usize>,
    header_addresses: Vec<u64>,
    header_lens: Vec<usize>,
    data_start: u64,
}

fn header_layout(plan: &Hdf5WritePlan, prepared: &[PreparedDataset<'_>]) -> Result<HeaderLayout> {
    let groups = plan_group_hierarchy(prepared, &plan.group_attributes)?;
    let zero_groups = vec![0; groups.paths.len()];
    let zero_datasets = vec![0; prepared.len()];
    let targets = placeholder_target_addresses(prepared, &groups, &zero_groups);
    let attributes = plan_attributes(plan, &groups, &targets)?;
    let root_len = encode_group_header_from_plan(
        &groups.root_links,
        &zero_groups,
        &zero_datasets,
        &attributes.root,
        0,
    )?
    .len();
    let mut next = align_u64(SUPERBLOCK_V2_SIZE as u64, 8);
    let root_address = next;
    next = align_u64(
        checked_add_u64(next, root_len as u64, "root header end")?,
        8,
    );
    let mut group_addresses = Vec::new();
    let mut group_lens = Vec::new();
    for (links, attrs) in groups.group_links.iter().zip(&attributes.groups) {
        let len =
            encode_group_header_from_plan(links, &zero_groups, &zero_datasets, attrs, 0)?.len();
        group_addresses.push(next);
        group_lens.push(len);
        next = align_u64(checked_add_u64(next, len as u64, "group header end")?, 8);
    }
    let mut header_addresses = Vec::new();
    let mut header_lens = Vec::new();
    for (dataset, attrs) in prepared.iter().zip(&attributes.datasets) {
        let len = encode_dataset_header(dataset, attrs, 0, 0)?.len();
        header_addresses.push(next);
        header_lens.push(len);
        next = align_u64(checked_add_u64(next, len as u64, "dataset header end")?, 8);
    }
    Ok(HeaderLayout {
        root_address,
        root_len,
        group_addresses,
        group_lens,
        header_addresses,
        header_lens,
        data_start: next,
    })
}

#[derive(Debug)]
struct PendingChunk {
    bytes: Vec<u8>,
    filled: u64,
}

#[derive(Debug)]
enum StreamKind {
    Contiguous {
        address: Option<u64>,
        size: u64,
    },
    Chunked {
        chunk_shape: Vec<u64>,
        /// Chunk grid position → (absolute address, stored size). A `BTreeMap` over the
        /// grid position iterates in row-major order, the B-tree key order.
        written: BTreeMap<Vec<u64>, (u64, u32)>,
        pending: BTreeMap<Vec<u64>, PendingChunk>,
    },
}

#[derive(Debug)]
struct DatasetStream {
    element_size: usize,
    filters: Vec<FilterDescription>,
    fill: Vec<u8>,
    kind: StreamKind,
}

/// Writes an [`Hdf5WritePlan`] whose [`DatasetBuilder::streamed`] datasets receive their data
/// after the file is laid out. See the module documentation.
pub struct Hdf5StreamWriter<W: Write + Seek> {
    sink: W,
    plan: Hdf5WritePlan,
    options: WriteOptions,
    layout: HeaderLayout,
    next_address: u64,
    streams: Vec<Option<DatasetStream>>,
}

impl<W: Write + Seek> Hdf5StreamWriter<W> {
    pub fn create(sink: W, plan: Hdf5WritePlan, options: WriteOptions) -> Result<Self> {
        if options.variant != Hdf5Variant::Modern {
            return Err(Error::UnsupportedFeature(format!(
                "unsupported HDF5 variant {:?}",
                options.variant
            )));
        }
        let (layout, streams) = {
            let prepared = prepare_datasets(&plan, options)?;
            let layout = header_layout(&plan, &prepared)?;
            let mut streams = Vec::with_capacity(prepared.len());
            for p in &prepared {
                let dataset = p.dataset;
                if !dataset.streamed {
                    streams.push(None);
                    continue;
                }
                let element_size = datatype_element_size(&dataset.datatype)?;
                let fill = match &dataset.fill_value {
                    Some(fill) => fill.clone(),
                    None => vec![0; element_size],
                };
                let kind = match &dataset.layout {
                    PlannedLayout::Chunked { chunk_shape } => StreamKind::Chunked {
                        chunk_shape: chunk_shape.clone(),
                        written: BTreeMap::new(),
                        pending: BTreeMap::new(),
                    },
                    _ => StreamKind::Contiguous {
                        address: None,
                        size: expected_data_len(&dataset.shape, element_size)? as u64,
                    },
                };
                streams.push(Some(DatasetStream {
                    element_size,
                    filters: p.filters.clone(),
                    fill,
                    kind,
                }));
            }
            (layout, streams)
        };
        let next_address = layout.data_start;
        Ok(Self {
            sink,
            plan,
            options,
            layout,
            next_address,
            streams,
        })
    }

    /// Index of the dataset with this path, as given to [`DatasetBuilder::new`].
    pub fn dataset_index(&self, name: &str) -> Option<usize> {
        self.plan.datasets.iter().position(|d| d.name == name)
    }

    pub fn shape(&self, dataset: usize) -> Option<&[u64]> {
        self.plan.datasets.get(dataset).map(|d| d.shape.as_slice())
    }

    fn stream_mut(&mut self, dataset: usize) -> Result<&mut DatasetStream> {
        let name = self
            .plan
            .datasets
            .get(dataset)
            .map(|d| d.name.clone())
            .ok_or_else(|| Error::InvalidDefinition(format!("no dataset {dataset}")))?;
        self.streams
            .get_mut(dataset)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::InvalidDefinition(format!("dataset '{name}' is not streamed")))
    }

    /// Grows a chunked streamed dataset along its resizable dimensions.
    pub fn set_extent(&mut self, dataset: usize, shape: &[u64]) -> Result<()> {
        let builder = self
            .plan
            .datasets
            .get(dataset)
            .ok_or_else(|| Error::InvalidDefinition(format!("no dataset {dataset}")))?;
        let max = builder
            .max_shape
            .clone()
            .unwrap_or_else(|| builder.shape.clone());
        if shape.len() != builder.shape.len()
            || !matches!(builder.layout, PlannedLayout::Chunked { .. })
        {
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' cannot be resized to {shape:?}",
                builder.name
            )));
        }
        for ((&new, &old), &max) in shape.iter().zip(&builder.shape).zip(&max) {
            if new < old || (max != UNLIMITED && new > max) {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' extent {shape:?} is outside {:?}..={max:?}",
                    builder.name, builder.shape
                )));
            }
        }
        self.stream_mut(dataset)?;
        self.plan.datasets[dataset].shape = shape.to_vec();
        Ok(())
    }

    /// Writes one whole chunk (`chunk` = its grid position; `data` = the full, unfiltered
    /// chunk in the dataset's byte order, edge chunks padded).
    pub fn write_chunk(&mut self, dataset: usize, chunk: &[u64], data: &[u8]) -> Result<()> {
        let shape = self.plan.datasets[dataset.min(self.plan.datasets.len().saturating_sub(1))]
            .shape
            .clone();
        let next = self.next_address;
        let stream = self.stream_mut(dataset)?;
        let StreamKind::Chunked {
            chunk_shape,
            written,
            pending,
        } = &mut stream.kind
        else {
            return Err(Error::InvalidDefinition(
                "write_chunk needs a chunked dataset".into(),
            ));
        };
        let chunk_bytes = chunk_byte_len(chunk_shape, stream.element_size)?;
        if data.len() != chunk_bytes || chunk.len() != shape.len() {
            return Err(Error::DataLengthMismatch {
                context: "streamed chunk".into(),
                expected: chunk_bytes,
                actual: data.len(),
            });
        }
        for ((&c, &dim), &cd) in chunk.iter().zip(&shape).zip(chunk_shape.iter()) {
            if c >= dim.div_ceil(cd).max(1) {
                return Err(Error::InvalidDefinition(format!(
                    "chunk {chunk:?} is outside the dataset extent {shape:?}"
                )));
            }
        }
        if written.contains_key(chunk) || pending.contains_key(chunk) {
            return Err(Error::InvalidDefinition(format!(
                "chunk {chunk:?} was already written"
            )));
        }
        let encoded = apply_write_filters(data, &stream.filters, stream.element_size)?;
        let size = u32::try_from(encoded.len()).map_err(|_| {
            Error::UnsupportedFeature("filtered chunk exceeds the 4-byte chunk size field".into())
        })?;
        written.insert(chunk.to_vec(), (next, size));
        self.sink.seek(SeekFrom::Start(next))?;
        self.sink.write_all(&encoded)?;
        self.next_address = checked_add_u64(next, encoded.len() as u64, "chunk end")?;
        Ok(())
    }

    /// Writes the box `start`..`start + count` (row-major `data` in the dataset's byte order).
    /// Chunks are emitted as soon as every element inside the dataset extent was written; each
    /// element should be written once.
    pub fn write_slice(
        &mut self,
        dataset: usize,
        start: &[u64],
        count: &[u64],
        data: &[u8],
    ) -> Result<()> {
        let shape = self
            .plan
            .datasets
            .get(dataset)
            .map(|d| d.shape.clone())
            .ok_or_else(|| Error::InvalidDefinition(format!("no dataset {dataset}")))?;
        let max_shape = self.plan.datasets[dataset]
            .max_shape
            .clone()
            .unwrap_or_else(|| shape.clone());
        let element_size = self.stream_mut(dataset)?.element_size;
        if start.len() != shape.len() || count.len() != shape.len() {
            return Err(Error::InvalidDefinition("slice rank mismatch".into()));
        }
        for ((&s, &c), &dim) in start.iter().zip(count).zip(&shape) {
            if s.checked_add(c).map_or(true, |end| end > dim) {
                return Err(Error::InvalidDefinition(format!(
                    "slice {start:?}+{count:?} is outside the dataset extent {shape:?}"
                )));
            }
        }
        let elements = count.iter().product::<u64>();
        if data.len() as u64 != elements * element_size as u64 {
            return Err(Error::DataLengthMismatch {
                context: "streamed slice".into(),
                expected: (elements * element_size as u64) as usize,
                actual: data.len(),
            });
        }
        if elements == 0 {
            return Ok(());
        }
        let is_chunked = matches!(self.stream_mut(dataset)?.kind, StreamKind::Chunked { .. });
        if is_chunked {
            self.write_chunked_slice(dataset, &shape, &max_shape, start, count, data)
        } else {
            self.write_contiguous_slice(dataset, &shape, start, count, data)
        }
    }

    fn write_contiguous_slice(
        &mut self,
        dataset: usize,
        shape: &[u64],
        start: &[u64],
        count: &[u64],
        data: &[u8],
    ) -> Result<()> {
        let next = self.next_address;
        let stream = self.stream_mut(dataset)?;
        let element_size = stream.element_size as u64;
        let StreamKind::Contiguous { address, size } = &mut stream.kind else {
            unreachable!()
        };
        let (base, size) = (*address, *size);
        let base = match base {
            Some(base) => base,
            None => {
                let base = align_u64(next, 8);
                *address = Some(base);
                let fill = stream.fill.clone();
                self.sink.seek(SeekFrom::Start(base))?;
                write_repeated(&mut self.sink, &fill, size)?;
                self.next_address = checked_add_u64(base, size, "contiguous data end")?;
                base
            }
        };
        if shape.is_empty() {
            // A scalar: its one element.
            self.sink.seek(SeekFrom::Start(base))?;
            self.sink.write_all(data)?;
            return Ok(());
        }
        let strides = row_major_strides(shape, "dataset stride")?;
        let row = count[count.len() - 1] * element_size;
        let mut source = 0usize;
        for_each_row(start, count, |coords| {
            let element: u64 = coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
            self.sink
                .seek(SeekFrom::Start(base + element * element_size))?;
            self.sink.write_all(&data[source..source + row as usize])?;
            source += row as usize;
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn write_chunked_slice(
        &mut self,
        dataset: usize,
        shape: &[u64],
        max_shape: &[u64],
        start: &[u64],
        count: &[u64],
        data: &[u8],
    ) -> Result<()> {
        let stream = self.stream_mut(dataset)?;
        let element_size = stream.element_size;
        let fill = stream.fill.clone();
        let StreamKind::Chunked {
            chunk_shape,
            written,
            pending,
        } = &mut stream.kind
        else {
            unreachable!()
        };
        let chunk_shape = chunk_shape.clone();
        let chunk_bytes = chunk_byte_len(&chunk_shape, element_size)?;
        let chunk_strides = row_major_strides(&chunk_shape, "chunk stride")?;
        let slice_strides = row_major_strides(count, "slice stride")?;
        let first: Vec<u64> = start.iter().zip(&chunk_shape).map(|(s, c)| s / c).collect();
        let last: Vec<u64> = start
            .iter()
            .zip(count)
            .zip(&chunk_shape)
            .map(|((s, n), c)| (s + n - 1) / c)
            .collect();
        let grid_count: Vec<u64> = first.iter().zip(&last).map(|(f, l)| l - f + 1).collect();
        let mut complete = Vec::new();
        let mut result = Ok(());
        for_each_index(&grid_count, |offset| {
            if result.is_err() {
                return;
            }
            let position: Vec<u64> = first.iter().zip(offset).map(|(f, o)| f + o).collect();
            if written.contains_key(&position) {
                result = Err(Error::InvalidDefinition(format!(
                    "chunk {position:?} was already written"
                )));
                return;
            }
            let chunk = pending
                .entry(position.clone())
                .or_insert_with(|| PendingChunk {
                    bytes: fill.iter().copied().cycle().take(chunk_bytes).collect(),
                    filled: 0,
                });
            // The part of this chunk inside the slice.
            let mut box_start = Vec::with_capacity(shape.len());
            let mut box_count = Vec::with_capacity(shape.len());
            for d in 0..shape.len() {
                let c0 = position[d] * chunk_shape[d];
                let lo = c0.max(start[d]);
                let hi = (c0 + chunk_shape[d]).min(start[d] + count[d]);
                box_start.push(lo);
                box_count.push(hi - lo);
            }
            let row_elements = box_count[shape.len() - 1] as usize;
            let row_bytes = row_elements * element_size;
            let _ = for_each_row(&box_start, &box_count, |coords| {
                let src: u64 = coords
                    .iter()
                    .zip(start)
                    .zip(&slice_strides)
                    .map(|((c, s), st)| (c - s) * st)
                    .sum();
                let dst: u64 = coords
                    .iter()
                    .zip(&position)
                    .zip(&chunk_shape)
                    .zip(&chunk_strides)
                    .map(|(((c, p), cs), st)| (c - p * cs) * st)
                    .sum();
                let (src, dst) = (src as usize * element_size, dst as usize * element_size);
                chunk.bytes[dst..dst + row_bytes].copy_from_slice(&data[src..src + row_bytes]);
                Ok(())
            });
            chunk.filled += box_count.iter().product::<u64>();
            // Complete when every element inside the extent is written; along resizable
            // dimensions the whole chunk length counts (more data may follow an extend).
            let needed: u64 = (0..shape.len())
                .map(|d| {
                    let c0 = position[d] * chunk_shape[d];
                    if max_shape[d] == UNLIMITED || max_shape[d] > shape[d] {
                        chunk_shape[d]
                    } else {
                        (shape[d] - c0).min(chunk_shape[d])
                    }
                })
                .product();
            if chunk.filled >= needed {
                complete.push(position);
            }
        });
        result?;
        for position in complete {
            let bytes = {
                let StreamKind::Chunked { pending, .. } = &mut self.stream_mut(dataset)?.kind
                else {
                    unreachable!()
                };
                pending.remove(&position).map(|c| c.bytes)
            };
            if let Some(bytes) = bytes {
                self.write_chunk(dataset, &position, &bytes)?;
            }
        }
        Ok(())
    }

    /// Emits the remaining data, the chunk indexes and the final headers.
    pub fn finish(mut self) -> Result<W> {
        // Chunks still in progress go out as they are (unwritten elements hold the fill).
        for dataset in 0..self.streams.len() {
            let pending = match self.streams[dataset].as_mut().map(|s| &mut s.kind) {
                Some(StreamKind::Chunked { pending, .. }) => std::mem::take(pending),
                _ => continue,
            };
            for (position, chunk) in pending {
                self.write_chunk(dataset, &position, &chunk.bytes)?;
            }
        }
        let plan = &self.plan;
        let mut prepared = prepare_datasets(plan, self.options)?;
        let mut streamed_addresses = vec![None; prepared.len()];
        for (index, (p, stream)) in prepared.iter_mut().zip(&self.streams).enumerate() {
            let Some(stream) = stream else { continue };
            match &stream.kind {
                StreamKind::Contiguous { address, size } => {
                    if let Some(address) = address {
                        p.data_size = *size;
                        streamed_addresses[index] = Some(*address);
                    }
                }
                StreamKind::Chunked {
                    chunk_shape,
                    written,
                    ..
                } => {
                    if !written.is_empty() {
                        p.chunk_index = Some(PreparedChunkIndex {
                            chunk_shape: chunk_shape.clone(),
                            chunks: written
                                .iter()
                                .map(|(position, &(address, size))| PreparedChunkIndexEntry {
                                    relative_address: address,
                                    size,
                                    filter_mask: 0,
                                    scaled_offsets: position.clone(),
                                })
                                .collect(),
                        });
                    }
                }
            }
        }
        let layout = header_layout(plan, &prepared)?;
        if layout != self.layout {
            return Err(Error::InvalidDefinition(
                "HDF5 header layout changed while streaming".into(),
            ));
        }
        let groups = plan_group_hierarchy(&prepared, &plan.group_attributes)?;
        let targets = target_addresses(
            &prepared,
            &layout.header_addresses,
            &groups,
            &layout.group_addresses,
        );
        let attributes = plan_attributes(plan, &groups, &targets)?;
        let heap = if attributes.heap_objects.is_empty() {
            Vec::new()
        } else {
            encode_global_heap_collection(&attributes.heap_objects)?
        };
        let data_start = align_u64(self.next_address, 8);
        let DataHeapLayout {
            data_addresses,
            heap_address,
            eof_address,
        } = stabilize_vlen_dataset_storage(&mut prepared, &attributes, data_start, &heap)?;

        // Chunk indexes after everything else (their size depends on the chunks written).
        let mut cursor = eof_address;
        let mut index_addresses = Vec::with_capacity(prepared.len());
        for p in &prepared {
            match &p.chunk_index {
                Some(index) => {
                    index_addresses.push(Some(cursor));
                    cursor = align_u64(
                        checked_add_u64(cursor, btree_v1_chunk_index_size(index)?, "index end")?,
                        8,
                    );
                }
                None => index_addresses.push(None),
            }
        }
        let eof = cursor;

        for (index, p) in prepared.iter().enumerate() {
            let streamed = self.streams[index].is_some();
            let data_address = streamed_addresses[index].unwrap_or(data_addresses[index]);
            if !streamed && !matches!(p.dataset.layout, PlannedLayout::Compact) {
                self.sink.seek(SeekFrom::Start(data_addresses[index]))?;
                self.sink.write_all(&p.raw_data)?;
            }
            if let (Some(chunk_index), Some(index_address)) =
                (&p.chunk_index, index_addresses[index])
            {
                let base = if streamed { 0 } else { data_addresses[index] };
                let nodes = encode_btree_v1_chunk_index(chunk_index, index_address, base)?;
                self.sink.seek(SeekFrom::Start(index_address))?;
                self.sink.write_all(&nodes)?;
            }
            let layout_address = index_addresses[index].unwrap_or(data_address);
            let header = encode_dataset_header(
                p,
                &attributes.datasets[index],
                layout_address,
                heap_address,
            )?;
            if header.len() != layout.header_lens[index] {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' header changed size while streaming",
                    p.dataset.name
                )));
            }
            self.sink
                .seek(SeekFrom::Start(layout.header_addresses[index]))?;
            self.sink.write_all(&header)?;
        }
        if !heap.is_empty() {
            self.sink.seek(SeekFrom::Start(heap_address))?;
            self.sink.write_all(&heap)?;
        }
        for (((links, attrs), &address), &len) in groups
            .group_links
            .iter()
            .zip(&attributes.groups)
            .zip(&layout.group_addresses)
            .zip(&layout.group_lens)
        {
            let header = encode_group_header_from_plan(
                links,
                &layout.group_addresses,
                &layout.header_addresses,
                attrs,
                heap_address,
            )?;
            if header.len() != len {
                return Err(Error::InvalidDefinition(
                    "group header changed size while streaming".into(),
                ));
            }
            self.sink.seek(SeekFrom::Start(address))?;
            self.sink.write_all(&header)?;
        }
        let root = encode_group_header_from_plan(
            &groups.root_links,
            &layout.group_addresses,
            &layout.header_addresses,
            &attributes.root,
            heap_address,
        )?;
        if root.len() != layout.root_len {
            return Err(Error::InvalidDefinition(
                "root header changed size while streaming".into(),
            ));
        }
        self.sink.seek(SeekFrom::Start(layout.root_address))?;
        self.sink.write_all(&root)?;
        self.sink.seek(SeekFrom::Start(0))?;
        self.sink
            .write_all(&encode_superblock_v2(layout.root_address, eof))?;
        // The file ends at the end-of-file address (alignment padding included).
        let end = self.sink.seek(SeekFrom::End(0))?;
        if end < eof {
            self.sink.write_all(&vec![0; (eof - end) as usize])?;
        }
        self.sink.flush()?;
        Ok(self.sink)
    }
}

fn chunk_byte_len(chunk_shape: &[u64], element_size: usize) -> Result<usize> {
    let elements = chunk_shape.iter().try_fold(1u64, |acc, &d| {
        checked_mul_u64(acc, d, "chunk element count")
    })?;
    checked_usize(
        checked_mul_u64(elements, element_size as u64, "chunk byte length")?,
        "chunk byte length",
    )
}

fn write_repeated(sink: &mut impl Write, pattern: &[u8], len: u64) -> Result<()> {
    let block: Vec<u8> = pattern.iter().copied().cycle().take(1 << 16).collect();
    let block = &block[..block.len() - block.len() % pattern.len().max(1)];
    let mut left = len as usize;
    while left > 0 {
        let n = left.min(block.len());
        sink.write_all(&block[..n])?;
        left -= n;
    }
    Ok(())
}

/// Calls `f` with the start coordinates of every row (run along the last dimension) of the
/// box `start`..`start + count`, in row-major order.
fn for_each_row(
    start: &[u64],
    count: &[u64],
    mut f: impl FnMut(&[u64]) -> Result<()>,
) -> Result<()> {
    let rank = start.len();
    if rank == 0 || count.contains(&0) {
        return Ok(());
    }
    let outer: Vec<u64> = count[..rank - 1].to_vec();
    let mut coords = start.to_vec();
    let mut result = Ok(());
    for_each_index(&outer, |offset| {
        if result.is_err() {
            return;
        }
        for d in 0..rank - 1 {
            coords[d] = start[d] + offset[d];
        }
        coords[rank - 1] = start[rank - 1];
        result = f(&coords);
    });
    result
}

/// Calls `f` with every index tuple below `counts`, row-major (once for rank 0).
fn for_each_index(counts: &[u64], mut f: impl FnMut(&[u64])) {
    if counts.contains(&0) {
        return;
    }
    let mut index = vec![0u64; counts.len()];
    loop {
        f(&index);
        let mut d = counts.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            index[d] += 1;
            if index[d] < counts[d] {
                break;
            }
            index[d] = 0;
        }
    }
}
