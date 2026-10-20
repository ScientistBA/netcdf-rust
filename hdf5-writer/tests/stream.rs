//! `Hdf5StreamWriter`: data written after the layout, chunk by chunk or slice by slice.

use std::io::Cursor;

use hdf5_reader::Hdf5File;
use hdf5_writer::{
    AttributeBuilder, DatasetBuilder, FilterDescription, Hdf5Builder, Hdf5StreamWriter, Hdf5Writer,
    WriteOptions, FILTER_DEFLATE, FILTER_SHUFFLE, UNLIMITED,
};

fn le_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn deflate() -> FilterDescription {
    FilterDescription {
        id: FILTER_DEFLATE,
        name: None,
        client_data: vec![4],
    }
}

fn shuffle() -> FilterDescription {
    FilterDescription {
        id: FILTER_SHUFFLE,
        name: None,
        client_data: vec![4],
    }
}

/// (t, y, x) field with edge chunks in y and x.
fn field() -> Vec<f32> {
    (0..4 * 5 * 6).map(|i| i as f32 * 0.5 - 7.0).collect()
}

#[test]
fn streamed_chunks_read_back_like_the_in_memory_file() {
    let values = field();
    let lat: Vec<f64> = (0..5).map(|i| 40.0 + i as f64).collect();
    let mask: Vec<i32> = (0..12).collect();
    let builder = |streamed: bool| {
        let hs = DatasetBuilder::typed::<f32>("hs", vec![4, 5, 6])
            .chunked(vec![1, 2, 4])
            .filter(shuffle())
            .filter(deflate())
            .fill_value((-999.0f32).to_le_bytes().to_vec())
            .attribute(AttributeBuilder::fixed_string("units", "m"));
        let grid = DatasetBuilder::typed::<i32>("mask", vec![3, 4]);
        let (hs, grid) = if streamed {
            (hs.streamed(), grid.streamed())
        } else {
            (hs.data(&values).unwrap(), grid.data(&mask).unwrap())
        };
        Hdf5Builder::new()
            .attribute(AttributeBuilder::fixed_string("title", "stream"))
            .dataset(DatasetBuilder::typed_data("lat", vec![5], &lat).unwrap())
            .dataset(hs)
            .dataset(grid)
            .dataset(
                DatasetBuilder::vlen_string_data("names", vec![2], &["north", "south"]).unwrap(),
            )
            .into_plan()
            .unwrap()
    };

    let mut writer = Hdf5StreamWriter::create(
        Cursor::new(Vec::new()),
        builder(true),
        WriteOptions::default(),
    )
    .unwrap();
    let hs = writer.dataset_index("hs").unwrap();
    let grid = writer.dataset_index("mask").unwrap();
    // One map per time step, out of order; the contiguous grid row by row, last row first.
    for t in [2u64, 0, 3, 1] {
        let plate = &values[(t as usize) * 30..(t as usize + 1) * 30];
        writer
            .write_slice(hs, &[t, 0, 0], &[1, 5, 6], &le_f32(plate))
            .unwrap();
    }
    for row in (0..3u64).rev() {
        let bytes: Vec<u8> = mask[row as usize * 4..row as usize * 4 + 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer
            .write_slice(grid, &[row, 0], &[1, 4], &bytes)
            .unwrap();
    }
    let streamed = writer.finish().unwrap().into_inner();
    let in_memory = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(builder(false))
        .unwrap()
        .into_inner();

    for bytes in [&streamed, &in_memory] {
        let file = Hdf5File::from_bytes(bytes).unwrap();
        let read = file.dataset("/hs").unwrap().read_array::<f32>().unwrap();
        assert_eq!(read.shape(), &[4, 5, 6]);
        assert_eq!(read.iter().copied().collect::<Vec<_>>(), values);
        let read = file.dataset("/mask").unwrap().read_array::<i32>().unwrap();
        assert_eq!(read.iter().copied().collect::<Vec<_>>(), mask);
        let read = file.dataset("/lat").unwrap().read_array::<f64>().unwrap();
        assert_eq!(read.iter().copied().collect::<Vec<_>>(), lat);
        assert_eq!(
            file.dataset("/hs")
                .unwrap()
                .attribute("units")
                .unwrap()
                .read_string()
                .unwrap(),
            "m"
        );
        assert_eq!(
            file.root_group()
                .unwrap()
                .attribute("title")
                .unwrap()
                .read_string()
                .unwrap(),
            "stream"
        );
        assert_eq!(
            file.dataset("/names").unwrap().read_strings().unwrap(),
            vec!["north".to_string(), "south".to_string()]
        );
    }
}

#[test]
fn unlimited_time_grows_and_unwritten_chunks_read_as_fill() {
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed::<f32>("series", vec![0, 3])
                .max_shape(vec![UNLIMITED, 3])
                .chunked(vec![2, 3])
                .filter(deflate())
                .fill_value((-1.0f32).to_le_bytes().to_vec())
                .streamed(),
        )
        .dataset(
            DatasetBuilder::typed::<f32>("sparse", vec![4, 4])
                .chunked(vec![2, 2])
                .fill_value((-5.0f32).to_le_bytes().to_vec())
                .streamed(),
        )
        .into_plan()
        .unwrap();
    let mut writer =
        Hdf5StreamWriter::create(Cursor::new(Vec::new()), plan, WriteOptions::default()).unwrap();
    let series = writer.dataset_index("series").unwrap();
    for t in 0..5u64 {
        writer.set_extent(series, &[t + 1, 3]).unwrap();
        let row = [t as f32, t as f32 + 0.25, t as f32 + 0.5];
        writer
            .write_slice(series, &[t, 0], &[1, 3], &le_f32(&row))
            .unwrap();
    }
    // Only the top-left chunk of `sparse`, through the whole-chunk API.
    let sparse = writer.dataset_index("sparse").unwrap();
    writer
        .write_chunk(sparse, &[0, 0], &le_f32(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    assert!(writer
        .write_chunk(sparse, &[0, 0], &le_f32(&[0.0; 4]))
        .is_err());
    let bytes = writer.finish().unwrap().into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let read = file
        .dataset("/series")
        .unwrap()
        .read_array::<f32>()
        .unwrap();
    assert_eq!(read.shape(), &[5, 3]);
    let expected: Vec<f32> = (0..5)
        .flat_map(|t| [t as f32, t as f32 + 0.25, t as f32 + 0.5])
        .collect();
    assert_eq!(read.iter().copied().collect::<Vec<_>>(), expected);
    let read = file
        .dataset("/sparse")
        .unwrap()
        .read_array::<f32>()
        .unwrap();
    assert_eq!(
        read.iter().copied().collect::<Vec<_>>(),
        vec![
            1.0, 2.0, -5.0, -5.0, //
            3.0, 4.0, -5.0, -5.0, //
            -5.0, -5.0, -5.0, -5.0, //
            -5.0, -5.0, -5.0, -5.0,
        ]
    );
}

#[test]
fn streaming_refuses_what_it_cannot_do() {
    let plan = Hdf5Builder::new()
        .dataset(DatasetBuilder::typed::<f32>("fixed", vec![2, 2]).streamed())
        .into_plan()
        .unwrap();
    let mut writer =
        Hdf5StreamWriter::create(Cursor::new(Vec::new()), plan, WriteOptions::default()).unwrap();
    let fixed = writer.dataset_index("fixed").unwrap();
    // Outside the extent, wrong length, resizing a contiguous dataset.
    assert!(writer
        .write_slice(fixed, &[1, 0], &[2, 2], &le_f32(&[0.0; 4]))
        .is_err());
    assert!(writer
        .write_slice(fixed, &[0, 0], &[1, 2], &le_f32(&[0.0; 3]))
        .is_err());
    assert!(writer.set_extent(fixed, &[3, 2]).is_err());
    let names = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::vlen_string_data("names", vec![1], &["a"])
                .unwrap()
                .streamed(),
        )
        .into_plan()
        .unwrap();
    assert!(
        Hdf5StreamWriter::create(Cursor::new(Vec::new()), names, WriteOptions::default()).is_err()
    );
}
