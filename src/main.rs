#![feature(core_intrinsics, ptr_internals)]
use fastrand::{self, usize};
use std::intrinsics::atomic_xchg_acqrel;
use std::iter::repeat_with;
use std::ptr;
use std::thread::spawn;
use std::{
    sync::atomic::{AtomicI8, Ordering},
    thread,
    time::Instant,
};
use threadpool::ThreadPool;

use nannou::{
    image::{DynamicImage, ImageBuffer, Luma},
    prelude::*,
};

use rand::distributions::{Distribution, Uniform};

use ndarray::Array2;
use ndarray_rand::RandomExt;

use itertools::izip;
use rayon::prelude::*;

const N: usize = 1000;

fn initialize_spin_lattice(dim: (usize, usize)) -> Array2<i8> {
    Array2::random(dim, Uniform::new(0., 1.)).mapv(|a| if a > 0.5 { 1 } else { -1 })
}

fn initialize_spin_lattice_atomic(dim: (usize, usize)) -> Array2<AtomicI8> {
    Array2::random(dim, Uniform::new(0., 1.)).mapv(|a| {
        if a > 0.5 {
            AtomicI8::new(1)
        } else {
            AtomicI8::new(-1)
        }
    })
}

// fn metropolis_step_parallel(lattice: &mut Array2<AtomicI8>, beta: f64) {
//     let n = lattice.shape()[0];
//     let m = lattice.shape()[1];
//
//     let x_values = Uniform::new(0, n).sample_iter(rand::thread_rng());
//     let y_values = Uniform::new(0, m).sample_iter(rand::thread_rng());
//     let r_values = Uniform::new(0., 1.).sample_iter(rand::thread_rng());
//
//     let changes: Vec<(usize, usize, f64)> = izip!(x_values, y_values, r_values).take(m * n).collect();
//     let mut handles = Vec::new();
//
//     changes.chunks((N*N)/8).clone().for_each(|vals| {
//         let handle = thread::spawn(move || {
//             for (x, y, r) in vals {
//                 let right = lattice[[(*x + 1) % n, *y]].load(Ordering::Relaxed);
//                 let down = lattice[[*x, (*y + 1) % m]].load(Ordering::Relaxed);
//                 let left = lattice[[(*x).saturating_sub(1) % n, *y]].load(Ordering::Relaxed);
//                 let up = lattice[[*x, (*y).saturating_sub(1) % m]].load(Ordering::Relaxed);
//                 let curr = lattice[[*x, *y]].load(Ordering::Relaxed);
//                 let neighbors_sum = right + down + left + up;
//                 let delta_e = (2 * lattice[[*x, *y]].load(Ordering::Relaxed) * neighbors_sum) as f64;
//                 if delta_e < 0_f64 || *r < (-beta * delta_e).exp() {
//                     lattice[[*x, *y]].store(-curr, Ordering::Relaxed);
//                 }
//             }
//         });
//         handles.push(handle);
//     });
//
//
// }

fn metropolis_step_atomic(lattice: &mut Array2<AtomicI8>, beta: f64) {
    let n = lattice.shape()[0];
    let m = lattice.shape()[1];

    let x_values = Uniform::new(0, n).sample_iter(rand::thread_rng());
    let y_values = Uniform::new(0, m).sample_iter(rand::thread_rng());
    let r_values = Uniform::new(0., 1.).sample_iter(rand::thread_rng());

    for (x, y, r) in izip!(x_values, y_values, r_values).take(m * n) {
        let right = lattice[[(x + 1) % n, y]].load(Ordering::Relaxed);
        let down = lattice[[x, (y + 1) % m]].load(Ordering::Relaxed);
        let left = lattice[[x.saturating_sub(1) % n, y]].load(Ordering::Relaxed);
        let up = lattice[[x, y.saturating_sub(1) % m]].load(Ordering::Relaxed);
        let curr = lattice[[x, y]].load(Ordering::Relaxed);
        let neighbors_sum = right + down + left + up;
        let delta_e = (2 * lattice[[x, y]].load(Ordering::Relaxed) * neighbors_sum) as f64;
        if delta_e < 0_f64 || r < (-beta * delta_e).exp() {
            lattice[[x, y]].store(-curr, Ordering::Relaxed);
        }
    }
}

fn metropolis_step(latt: *mut i8, beta: f64) {
    let n = N;
    let m = N;
    unsafe {
        let x_values = repeat_with(|| fastrand::Rng::new().usize(0..n));
        let y_values = repeat_with(|| fastrand::Rng::new().usize(0..n));
        let r_values = repeat_with(|| fastrand::Rng::new().f64());
        for (x, y, r) in izip!(x_values, y_values, r_values).take(m * n) {
            let curr = latt.add(x + (m * y));
            let right = latt.add(((x + 1) % n) + (m * y));
            let down = latt.add(x + (m * ((y + 1) % m)));
            let left = latt.add((x.saturating_sub(1) % n) + (m * y));
            let up = latt.add(x + (m * (y.saturating_sub(1) % m)));
            let neighbors_sum = *right + *down + *left + *up;
            let delta_e = (2 * *curr * neighbors_sum) as f64;
            if delta_e < 0_f64 || r < (-beta * delta_e).exp() {
                atomic_xchg_acqrel(curr, *curr * -1);
            }
        }
    };
}

fn main() {
    nannou::app(model).update(update).simple_window(view).run();
}

struct Model {
    pool: ThreadPool,
    lattice: Array2<i8>,
}

fn model(_app: &App) -> Model {
    Model {
        pool: ThreadPool::new(std::thread::available_parallelism().unwrap().get()),
        lattice: initialize_spin_lattice((N, N)),
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    let beta = (_app.mouse.x as f64 + ((N / 2) as f64)) / ((N / 2) as f64);
    let start = Instant::now();
    let data_ptr = ptr::Unique::new(_model.lattice.as_mut_ptr());
    _model
        .pool
        .execute(move || metropolis_step(data_ptr.unwrap().as_ptr(), beta));

    let duration = start.elapsed();
    print!("\rTime elapsed in iteration is: {:?}", duration);
}

fn view(_app: &App, _model: &Model, frame: Frame) {
    let draw = _app.draw();
    let vec: Vec<u8> = _model
        .lattice
        .iter()
        .map(|x| u8::try_from(255 * (*x as i16 + 1) / 2).unwrap())
        .collect();
    // let vec: Vec<u8> = _model.lattice.clone().into_raw_vec().iter().map(|x| u8::try_from(255*(*x as i16 + 1)/2).unwrap()).collect();
    let image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(
        _model.lattice.shape()[0] as u32,
        _model.lattice.shape()[1] as u32,
        vec,
    )
    .unwrap();
    let dynamic_image: DynamicImage = DynamicImage::ImageLuma8(image);
    let texture = wgpu::Texture::from_image(_app, &dynamic_image);
    draw.background().color(BLACK);
    draw.texture(&texture);
    draw.to_frame(_app, &frame).unwrap();
}
