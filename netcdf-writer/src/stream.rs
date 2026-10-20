//! Streaming NetCDF writing: define the file, then write each variable slice by slice.
//!
//! ```no_run
//! # use netcdf_writer::{NcFileBuilder, NcWriteOptions};
//! let mut b = NcFileBuilder::new();
//! let t = b.add_unlimited_dimension("time")?;
//! let y = b.add_dimension("y", 180)?;
//! let x = b.add_dimension("x", 360)?;
//! let hs = b.add_variable::<f32>("hs", &[t, y, x])?;
//! b.set_variable_streamed(hs)?;
//! let file = std::fs::File::create("out.nc")?;
//! let mut w = b.stream(file, NcWriteOptions::offset64())?;
//! for step in 0..24u64 {
//!     let plate = vec![0.0f32; 180 * 360];
//!     w.write_slice(hs, &[step, 0, 0], &[1, 180, 360], &plate)?;
//! }
//! w.finish()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The definitions (dimensions, attributes, variables) are final when streaming starts; only
//! the length of an unlimited dimension grows as records are written. Variables marked with
//! [`NcFileBuilder::set_variable_streamed`] get their data through
//! [`NcStreamWriter::write_slice`]; the others are written from the builder as usual (variables
//! along an unlimited dimension must be streamed).
//!
//! - **Classic** (CDF-1/2/5): the header goes first and every slice to its computed offset, in
//!   any order. With fill (the default) a variable's space — a record's, for record variables —
//!   is set to the fill value when it is first touched; [`NcFileBuilder::stream_without_fill`]
//!   skips that and instead requires every element to be written. The record count is written
//!   at `finish`.
//! - **NetCDF-4**: through `hdf5_writer::Hdf5StreamWriter` — headers first, each chunk filtered
//!   and appended as soon as it is complete, indexes and final headers at `finish`; elements
//!   never written read as the fill value.
//!
//! Memory: the definitions, the non-streamed data and the NetCDF-4 chunks in progress.

use super::*;
use std::io::{Seek, SeekFrom};

#[cfg(feature = "netcdf4")]
use hdf5_writer::Hdf5StreamWriter;

pub(crate) fn is_streamable(dtype: &NcType) -> bool {
    matches!(
        dtype,
        NcType::Byte
            | NcType::UByte
            | NcType::Short
            | NcType::UShort
            | NcType::Int
            | NcType::UInt
            | NcType::Int64
            | NcType::UInt64
            | NcType::Float
            | NcType::Double
    )
}

impl NcFileBuilder {
    /// Starts streaming the file to `sink`; see [`NcStreamWriter`].
    pub fn stream<W: Write + Seek>(
        self,
        sink: W,
        options: NcWriteOptions,
    ) -> Result<NcStreamWriter<W>> {
        NcStreamWriter::create(self, sink, options, true)
    }

    /// As [`NcFileBuilder::stream`], without pre-filling classic variables with the fill value:
    /// faster, and `finish` fails unless every element of every streamed variable was written.
    pub fn stream_without_fill<W: Write + Seek>(
        self,
        sink: W,
        options: NcWriteOptions,
    ) -> Result<NcStreamWriter<W>> {
        NcStreamWriter::create(self, sink, options, false)
    }
}

#[derive(Debug)]
struct ClassicVar {
    begin: u64,
    is_record: bool,
    layout_vsize: u64,
    /// Bytes of data per record (record variables) or in total.
    data_len: u64,
    fill: Vec<u8>,
}

enum Inner<W: Write + Seek> {
    Classic {
        sink: W,
        vars: Vec<ClassicVar>,
        record_stride: u64,
        prefill: bool,
        filled_records: u64,
    },
    #[cfg(feature = "netcdf4")]
    Nc4 {
        writer: Box<Hdf5StreamWriter<W>>,
        /// HDF5 dataset of each streamed variable.
        datasets: Vec<Option<usize>>,
        /// Hidden dimension-scale datasets of unlimited dimensions: (dimension, dataset).
        scales: Vec<(usize, usize)>,
    },
}

/// A NetCDF file being written slice by slice; made by [`NcFileBuilder::stream`].
pub struct NcStreamWriter<W: Write + Seek> {
    defs: NcFileBuilder,
    format: NcFormat,
    /// Current length of every dimension (unlimited ones grow).
    lengths: Vec<u64>,
    written: Vec<u64>,
    inner: Inner<W>,
}

impl<W: Write + Seek> NcStreamWriter<W> {
    fn create(
        defs: NcFileBuilder,
        mut sink: W,
        options: NcWriteOptions,
        prefill: bool,
    ) -> Result<Self> {
        for variable in &defs.variables {
            if !variable.streamed
                && variable
                    .dim_ids
                    .iter()
                    .any(|id| defs.dimensions[id.0].is_unlimited)
            {
                return Err(Error::InvalidDefinition(format!(
                    "variable '{}' uses an unlimited dimension and must be streamed",
                    variable.name
                )));
            }
        }
        let format = match options.format {
            NcWriteFormat::AutoClassic => {
                if defs.requires_cdf5() {
                    NcFormat::Cdf5
                } else if ClassicWritePlan::build(&defs, NcFormat::Classic).is_ok() {
                    NcFormat::Classic
                } else {
                    NcFormat::Offset64
                }
            }
            _ => defs.select_format(options)?,
        };
        let lengths: Vec<u64> = defs
            .dimensions
            .iter()
            .map(|d| if d.is_unlimited { 0 } else { d.size })
            .collect();
        let inner = match format {
            NcFormat::Classic | NcFormat::Offset64 | NcFormat::Cdf5 => {
                let plan = ClassicWritePlan::build(&defs, format)?;
                sink.seek(SeekFrom::Start(0))?;
                sink.write_all(&plan.header)?;
                let mut vars = Vec::with_capacity(defs.variables.len());
                for def in &defs.variables {
                    let planned = plan
                        .fixed_vars
                        .iter()
                        .chain(&plan.record_vars)
                        .find(|p| p.def.name == def.name)
                        .ok_or_else(|| {
                            Error::InvalidDefinition(format!("variable '{}' not planned", def.name))
                        })?;
                    let fill = variable_fill_pattern(def)?;
                    let var = ClassicVar {
                        begin: planned.begin,
                        is_record: planned.is_record,
                        layout_vsize: planned.layout_vsize,
                        data_len: if planned.is_record {
                            planned.record_slab_len
                        } else {
                            planned.fixed_data_len
                        },
                        fill,
                    };
                    if !var.is_record {
                        sink.seek(SeekFrom::Start(var.begin))?;
                        if def.streamed {
                            let len = if prefill { var.layout_vsize } else { 0 };
                            write_fill(&mut sink, len, &var.fill)?;
                            if !prefill {
                                sink.seek(SeekFrom::Start(var.begin + var.data_len))?;
                                write_fill_padding(
                                    &mut sink,
                                    var.layout_vsize - var.data_len,
                                    &var.fill,
                                )?;
                            }
                        } else {
                            sink.write_all(&def.data)?;
                            write_fill_padding(
                                &mut sink,
                                var.layout_vsize - var.data_len,
                                &var.fill,
                            )?;
                        }
                    }
                    vars.push(var);
                }
                let record_stride = vars
                    .iter()
                    .filter(|v| v.is_record)
                    .map(|v| v.layout_vsize)
                    .sum();
                Inner::Classic {
                    sink,
                    vars,
                    record_stride,
                    prefill,
                    filled_records: 0,
                }
            }
            #[cfg(feature = "netcdf4")]
            NcFormat::Nc4 | NcFormat::Nc4Classic => {
                defs.validate_for_nc4_bridge(format)?;
                let plan = defs.build_hdf5_plan_with(format, &lengths, true)?;
                let writer = Hdf5StreamWriter::create(sink, plan, H5WriteOptions::default())
                    .map_err(hdf5_error_to_unsupported)?;
                let datasets = defs
                    .variables
                    .iter()
                    .map(|v| v.streamed.then(|| writer.dataset_index(&v.name)).flatten())
                    .collect();
                let scales = defs
                    .dimensions
                    .iter()
                    .enumerate()
                    .filter(|(id, d)| {
                        d.is_unlimited
                            && defs
                                .coordinate_variable_for_dimension(DimensionId(*id))
                                .is_none()
                    })
                    .filter_map(|(id, d)| writer.dataset_index(&d.name).map(|ds| (id, ds)))
                    .collect();
                Inner::Nc4 {
                    writer: Box::new(writer),
                    datasets,
                    scales,
                }
            }
            #[cfg(not(feature = "netcdf4"))]
            NcFormat::Nc4 | NcFormat::Nc4Classic => {
                return Err(Error::UnsupportedFeature(
                    "NetCDF-4 writing requires the netcdf4 feature".into(),
                ))
            }
        };
        let written = vec![0; defs.variables.len()];
        Ok(Self {
            defs,
            format,
            lengths,
            written,
            inner,
        })
    }

    /// The format being written (`AutoClassic` resolved).
    pub fn format(&self) -> NcFormat {
        self.format
    }

    /// Current length of a dimension (an unlimited one grows as records are written).
    pub fn dimension_length(&self, dimension: DimensionId) -> Option<u64> {
        self.lengths.get(dimension.0).copied()
    }

    /// Writes the box `start`..`start + count` of a streamed variable (`values` row-major).
    /// Writing past the end of an unlimited first dimension extends it.
    pub fn write_slice<T: NcWriteType>(
        &mut self,
        variable: VariableId,
        start: &[u64],
        count: &[u64],
        values: &[T],
    ) -> Result<()> {
        let def = self.defs.variable(variable)?;
        if !def.streamed {
            return Err(Error::InvalidDefinition(format!(
                "variable '{}' is not streamed",
                def.name
            )));
        }
        if T::nc_type() != def.dtype {
            return Err(Error::InvalidDefinition(format!(
                "variable '{}' is {:?}, not {:?}",
                def.name,
                def.dtype,
                T::nc_type()
            )));
        }
        let rank = def.dim_ids.len();
        if start.len() != rank || count.len() != rank {
            return Err(Error::InvalidDefinition(format!(
                "slice rank does not match variable '{}' ({rank} dimensions)",
                def.name
            )));
        }
        let elements = count.iter().try_fold(1u64, |acc, &c| {
            acc.checked_mul(c)
                .ok_or_else(|| Error::InvalidDefinition("slice element count overflows".into()))
        })?;
        if values.len() as u64 != elements {
            return Err(Error::DataLengthMismatch {
                expected: elements as usize,
                actual: values.len(),
            });
        }
        let mut extents = Vec::with_capacity(rank);
        let mut grows = None;
        for (axis, (&s, &c)) in start.iter().zip(count).enumerate() {
            let dim_id = def.dim_ids[axis].0;
            let dim = &self.defs.dimensions[dim_id];
            let end = s
                .checked_add(c)
                .ok_or_else(|| Error::InvalidDefinition("slice end overflows".into()))?;
            if dim.is_unlimited {
                if end > self.lengths[dim_id] {
                    grows = Some((dim_id, end));
                }
                extents.push(end.max(self.lengths[dim_id]));
            } else {
                if end > dim.size {
                    return Err(Error::InvalidDefinition(format!(
                        "slice {start:?}+{count:?} is outside variable '{}'",
                        def.name
                    )));
                }
                extents.push(dim.size);
            }
        }
        if elements == 0 {
            return Ok(());
        }
        let dims: Vec<DimensionId> = def.dim_ids.clone();
        if let Some((dim_id, end)) = grows {
            self.grow(dim_id, end)?;
        }
        let var_index = variable.0;
        match &mut self.inner {
            Inner::Classic {
                sink,
                vars,
                record_stride,
                prefill,
                filled_records,
            } => {
                let mut bytes = Vec::with_capacity(values.len() * T::nc_type().size()?);
                for &value in values {
                    value.write_one_be(&mut bytes);
                }
                let elem = (bytes.len() / values.len()) as u64;
                let var = &vars[var_index];
                if var.is_record {
                    let last = start[0] + count[0];
                    if *prefill && last > *filled_records {
                        for record in *filled_records..last {
                            for v in vars.iter().filter(|v| v.is_record) {
                                sink.seek(SeekFrom::Start(v.begin + record * *record_stride))?;
                                write_fill(sink, v.layout_vsize, &v.fill)?;
                            }
                        }
                        *filled_records = last;
                    }
                    let var = &vars[var_index];
                    let inner_shape = &extents[1..];
                    let strides = strides(inner_shape);
                    let per_record = (bytes.len() as u64) / count[0];
                    for r in 0..count[0] {
                        let record = start[0] + r;
                        let base = var.begin + record * *record_stride;
                        let chunk =
                            &bytes[(r * per_record) as usize..((r + 1) * per_record) as usize];
                        write_box(sink, base, elem, &strides, &start[1..], &count[1..], chunk)?;
                    }
                } else {
                    let strides = strides(&extents);
                    write_box(sink, var.begin, elem, &strides, start, count, &bytes)?;
                }
            }
            #[cfg(feature = "netcdf4")]
            Inner::Nc4 {
                writer, datasets, ..
            } => {
                let mut bytes = Vec::with_capacity(values.len() * T::nc_type().size()?);
                for &value in values {
                    value.write_one_le(&mut bytes);
                }
                let dataset = datasets[var_index].ok_or_else(|| {
                    Error::InvalidDefinition("streamed variable has no dataset".into())
                })?;
                writer
                    .write_slice(dataset, start, count, &bytes)
                    .map_err(hdf5_error_to_unsupported)?;
            }
        }
        let _ = dims;
        self.written[var_index] += elements;
        Ok(())
    }

    fn grow(&mut self, dim_id: usize, end: u64) -> Result<()> {
        self.lengths[dim_id] = end;
        #[cfg(feature = "netcdf4")]
        if let Inner::Nc4 {
            writer,
            datasets,
            scales,
        } = &mut self.inner
        {
            for (variable, dataset) in self.defs.variables.iter().zip(datasets.iter()) {
                let Some(dataset) = dataset else { continue };
                if !variable.dim_ids.iter().any(|id| id.0 == dim_id) {
                    continue;
                }
                let shape: Vec<u64> = variable
                    .dim_ids
                    .iter()
                    .map(|id| self.lengths[id.0])
                    .collect();
                writer
                    .set_extent(*dataset, &shape)
                    .map_err(hdf5_error_to_unsupported)?;
            }
            for &(dim, dataset) in scales.iter() {
                if dim == dim_id {
                    writer
                        .set_extent(dataset, &[end])
                        .map_err(hdf5_error_to_unsupported)?;
                }
            }
        }
        Ok(())
    }

    /// Completes the file and returns the sink.
    pub fn finish(self) -> Result<W> {
        let Self {
            defs,
            format,
            lengths,
            written,
            inner,
        } = self;
        match inner {
            Inner::Classic {
                mut sink,
                vars,
                record_stride,
                prefill,
                filled_records,
            } => {
                let records = defs
                    .dimensions
                    .iter()
                    .enumerate()
                    .find(|(_, d)| d.is_unlimited)
                    .map(|(id, _)| lengths[id])
                    .unwrap_or(0);
                for (def, (var, &count)) in defs.variables.iter().zip(vars.iter().zip(&written)) {
                    if !def.streamed {
                        continue;
                    }
                    let elem = def.dtype.size()? as u64;
                    let needed = if var.is_record {
                        records * (var.data_len / elem.max(1))
                    } else {
                        var.data_len / elem.max(1)
                    };
                    if !prefill && count < needed {
                        return Err(Error::InvalidDefinition(format!(
                            "variable '{}' has {count} of {needed} elements written (streaming without fill)",
                            def.name
                        )));
                    }
                }
                // Record padding, and records never touched by a variable, with fill.
                for record in 0..records {
                    for var in vars.iter().filter(|v| v.is_record) {
                        let base = var.begin + record * record_stride;
                        if prefill && record >= filled_records {
                            sink.seek(SeekFrom::Start(base))?;
                            write_fill(&mut sink, var.layout_vsize, &var.fill)?;
                        } else if var.layout_vsize > var.data_len {
                            sink.seek(SeekFrom::Start(base + var.data_len))?;
                            write_fill_padding(
                                &mut sink,
                                var.layout_vsize - var.data_len,
                                &var.fill,
                            )?;
                        }
                    }
                }
                let mut count = Vec::new();
                write_count(&mut count, format, records)?;
                sink.seek(SeekFrom::Start(4))?;
                sink.write_all(&count)?;
                sink.flush()?;
                Ok(sink)
            }
            #[cfg(feature = "netcdf4")]
            Inner::Nc4 {
                mut writer, scales, ..
            } => {
                for (dim, dataset) in scales {
                    let len = lengths[dim];
                    if len > 0 {
                        writer
                            .write_slice(dataset, &[0], &[len], &vec![0u8; len as usize * 4])
                            .map_err(hdf5_error_to_unsupported)?;
                    }
                }
                let _ = written;
                writer.finish().map_err(hdf5_error_to_unsupported)
            }
        }
    }
}

/// `len` bytes of the repeated fill pattern (whole variables and records; the classic
/// `write_fill_padding` covers only the up-to-3-byte alignment padding).
fn write_fill(sink: &mut impl Write, len: u64, pattern: &[u8]) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    let pattern: &[u8] = if pattern.is_empty() { &[0] } else { pattern };
    let block: Vec<u8> = pattern
        .iter()
        .copied()
        .cycle()
        .take(pattern.len() * 8192)
        .collect();
    let mut left = len;
    while left > 0 {
        let n = left.min(block.len() as u64) as usize;
        sink.write_all(&block[..n])?;
        left -= n as u64;
    }
    Ok(())
}

fn strides(shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Writes a row-major box of big-endian bytes into a variable laid out with `strides`.
fn write_box<W: Write + Seek>(
    sink: &mut W,
    base: u64,
    elem: u64,
    strides: &[u64],
    start: &[u64],
    count: &[u64],
    bytes: &[u8],
) -> Result<()> {
    let rank = start.len();
    if rank == 0 {
        sink.seek(SeekFrom::Start(base))?;
        sink.write_all(bytes)?;
        return Ok(());
    }
    let row = (count[rank - 1] * elem) as usize;
    let rows: u64 = count[..rank - 1].iter().product();
    let mut index = vec![0u64; rank - 1];
    for r in 0..rows {
        let element: u64 = (0..rank - 1)
            .map(|d| (start[d] + index[d]) * strides[d])
            .sum::<u64>()
            + start[rank - 1] * strides[rank - 1];
        sink.seek(SeekFrom::Start(base + element * elem))?;
        let from = r as usize * row;
        sink.write_all(&bytes[from..from + row])?;
        for d in (0..rank - 1).rev() {
            index[d] += 1;
            if index[d] < count[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(())
}
