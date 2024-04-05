use std::iter::repeat_with;
use std::time::Instant;
use threadpool::ThreadPool;

use nannou::{
    image::{DynamicImage, ImageBuffer, Luma},
    prelude::*,
};

use rand::distributions::{Distribution, Uniform};

use ndarray::Array2;
use ndarray_rand::RandomExt;

use itertools::izip;

const N: usize = 1000;

fn initialize_spin_lattice(dim: (usize, usize)) -> Array2<i8> {
    Array2::random(dim, Uniform::new(0., 1.)).mapv(|a| if a > 0.5 { 1 } else { -1 })
}
struct HoldsRawPtr {
    ptr: *mut i8,
}
unsafe impl Send for HoldsRawPtr {}
// Sync trait tells compiler that `&T` is safe to share between threads
unsafe impl Sync for HoldsRawPtr {}

fn metropolis_step(latt: HoldsRawPtr, beta: f64) {
    let n = N;
    let m = N;
    let latt = latt.ptr;
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
                *curr = *curr * -1;
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
    let data_ptr = HoldsRawPtr {
        ptr: (_model.lattice.as_mut_ptr()),
    };
    _model.pool.execute(move || metropolis_step(data_ptr, beta));

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
