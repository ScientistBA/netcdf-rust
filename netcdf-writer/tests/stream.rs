//! `NcStreamWriter`: files defined first and filled slice by slice.

use std::io::Cursor;

use netcdf_reader::NcFile;
#[cfg(feature = "netcdf4")]
use netcdf_writer::NcWriteFormat;
use netcdf_writer::{NcAttrValue, NcFileBuilder, NcWriteOptions};

const T: u64 = 4;
const Y: u64 = 3;
const X: u64 = 5;

fn hs_value(t: u64, y: u64, x: u64) -> f32 {
    (t * 100 + y * 10 + x) as f32 * 0.25
}

/// Builds the same file either streamed or in memory: an unlimited time axis with its
/// coordinate, a static latitude, a (time, y, x) field and a fixed (y, x) depth.
fn define(streamed: bool) -> (NcFileBuilder, [netcdf_writer::VariableId; 4]) {
    let mut b = NcFileBuilder::new();
    let t = b.add_unlimited_dimension("time").unwrap();
    let y = b.add_dimension("lat", Y).unwrap();
    let x = b.add_dimension("lon", X).unwrap();
    b.add_attribute("Conventions", NcAttrValue::Chars("CF-1.10".into()))
        .unwrap();
    let time = b.add_variable::<f64>("time", &[t]).unwrap();
    b.add_variable_attribute(
        time,
        "units",
        NcAttrValue::Chars("hours since 2026-01-01".into()),
    )
    .unwrap();
    let lat = b.add_variable::<f64>("lat", &[y]).unwrap();
    b.write_variable(lat, &[40.0, 41.0, 42.0]).unwrap();
    let hs = b.add_variable::<f32>("hs", &[t, y, x]).unwrap();
    b.set_variable_fill_value(hs, -999.0f32).unwrap();
    let depth = b.add_variable::<i16>("depth", &[y, x]).unwrap();
    if streamed {
        for v in [time, hs, depth] {
            b.set_variable_streamed(v).unwrap();
        }
    } else {
        b.write_variable(time, &(0..T).map(|t| t as f64 * 6.0).collect::<Vec<_>>())
            .unwrap();
        let all: Vec<f32> = (0..T)
            .flat_map(|t| (0..Y).flat_map(move |y| (0..X).map(move |x| hs_value(t, y, x))))
            .collect();
        b.write_variable(hs, &all).unwrap();
        b.write_variable(depth, &(0..(Y * X) as i16).collect::<Vec<_>>())
            .unwrap();
    }
    (b, [time, lat, hs, depth])
}

fn stream_file(options: NcWriteOptions, fill: bool) -> Vec<u8> {
    let (b, [time, _, hs, depth]) = define(true);
    let sink = Cursor::new(Vec::new());
    let mut w = if fill {
        b.stream(sink, options).unwrap()
    } else {
        b.stream_without_fill(sink, options).unwrap()
    };
    // Time steps out of order; each step's coordinate after its field.
    for t in [1u64, 0, 3, 2] {
        let plate: Vec<f32> = (0..Y)
            .flat_map(|y| (0..X).map(move |x| hs_value(t, y, x)))
            .collect();
        w.write_slice(hs, &[t, 0, 0], &[1, Y, X], &plate).unwrap();
        w.write_slice(time, &[t], &[1], &[t as f64 * 6.0]).unwrap();
    }
    // The depth grid column by column.
    for x in 0..X {
        let column: Vec<i16> = (0..Y).map(|y| (y * X + x) as i16).collect();
        w.write_slice(depth, &[0, x], &[Y, 1], &column).unwrap();
    }
    w.finish().unwrap().into_inner()
}

#[test]
fn streamed_classic_files_equal_the_in_memory_files_byte_for_byte() {
    for options in [
        NcWriteOptions::classic(),
        NcWriteOptions::offset64(),
        NcWriteOptions::cdf5(),
    ] {
        let (b, _) = define(false);
        let (_, expected) = b.to_vec(options).unwrap();
        assert_eq!(
            stream_file(options, true),
            expected,
            "{options:?} with fill"
        );
        assert_eq!(
            stream_file(options, false),
            expected,
            "{options:?} without fill"
        );
    }
}

#[cfg(feature = "netcdf4")]
#[test]
fn streamed_netcdf4_reads_back_like_the_in_memory_file() {
    let options = NcWriteOptions {
        format: NcWriteFormat::Nc4,
    };
    let streamed = stream_file(options, true);
    let (b, _) = define(false);
    let (_, in_memory) = b.to_vec(options).unwrap();
    for bytes in [&streamed, &in_memory] {
        let file = NcFile::from_bytes(bytes).unwrap();
        let time = file.dimension("time").unwrap();
        assert!(time.is_unlimited);
        assert_eq!(time.size, T);
        let hs = file.read_variable::<f32>("hs").unwrap();
        assert_eq!(hs.shape(), &[T as usize, Y as usize, X as usize]);
        for t in 0..T {
            for y in 0..Y {
                for x in 0..X {
                    assert_eq!(hs[[t as usize, y as usize, x as usize]], hs_value(t, y, x));
                }
            }
        }
        let times = file.read_variable::<f64>("time").unwrap();
        assert_eq!(
            times.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 6.0, 12.0, 18.0]
        );
        let depth = file.read_variable::<i16>("depth").unwrap();
        assert_eq!(
            depth.iter().copied().collect::<Vec<_>>(),
            (0..(Y * X) as i16).collect::<Vec<_>>()
        );
        assert_eq!(
            file.variable("time")
                .unwrap()
                .attribute("units")
                .unwrap()
                .value
                .as_string()
                .unwrap(),
            "hours since 2026-01-01"
        );
    }
}

#[test]
fn streaming_without_fill_refuses_unwritten_elements_and_fill_covers_them() {
    let partial = |fill: bool| {
        let (b, [time, _, hs, depth]) = define(true);
        let sink = Cursor::new(Vec::new());
        let mut w = if fill {
            b.stream(sink, NcWriteOptions::offset64()).unwrap()
        } else {
            b.stream_without_fill(sink, NcWriteOptions::offset64())
                .unwrap()
        };
        w.write_slice(time, &[0], &[2], &[0.0, 6.0]).unwrap();
        // Only record 1 of the field, and no depth at all.
        w.write_slice(hs, &[1, 0, 0], &[1, Y, X], &vec![1.0f32; (Y * X) as usize])
            .unwrap();
        let _ = depth;
        w.finish().map(Cursor::into_inner)
    };
    assert!(partial(false).is_err());
    let bytes = partial(true).unwrap();
    let file = NcFile::from_bytes(&bytes).unwrap();
    let hs = file.read_variable::<f32>("hs").unwrap();
    assert_eq!(hs.shape(), &[2, Y as usize, X as usize]);
    assert!(hs.iter().take((Y * X) as usize).all(|&v| v == -999.0));
    assert!(hs.iter().skip((Y * X) as usize).all(|&v| v == 1.0));
    let depth = file.read_variable::<i16>("depth").unwrap();
    assert!(depth.iter().all(|&v| v == -32767), "{depth:?}");
}

#[test]
fn streaming_refuses_bad_slices_and_unstreamed_record_variables() {
    let (b, [time, lat, hs, _]) = define(true);
    let mut w = b
        .stream(Cursor::new(Vec::new()), NcWriteOptions::offset64())
        .unwrap();
    assert!(
        w.write_slice(lat, &[0], &[3], &[0.0f64; 3]).is_err(),
        "not streamed"
    );
    assert!(w
        .write_slice(
            hs,
            &[0, 0, 0],
            &[1, Y + 1, X],
            &vec![0.0f32; ((Y + 1) * X) as usize]
        )
        .is_err());
    assert!(
        w.write_slice(hs, &[0, 0, 0], &[1, Y, X], &vec![0.0f64; (Y * X) as usize])
            .is_err(),
        "wrong type"
    );
    assert!(
        w.write_slice(time, &[0], &[2], &[0.0f64]).is_err(),
        "wrong length"
    );

    let mut b = NcFileBuilder::new();
    let t = b.add_unlimited_dimension("time").unwrap();
    let v = b.add_variable::<f32>("v", &[t]).unwrap();
    b.write_variable(v, &[1.0f32, 2.0]).unwrap();
    assert!(b
        .stream(Cursor::new(Vec::new()), NcWriteOptions::offset64())
        .is_err());
}
