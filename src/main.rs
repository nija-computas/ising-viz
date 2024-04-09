use std::{
    iter::repeat_with,
    sync::{
        atomic::{AtomicI8, Ordering},
        Arc,
    },
};
use threadpool::ThreadPool;

use nannou::{
    image::{DynamicImage, ImageBuffer, Luma},
    prelude::*,
};

use rand::distributions::Uniform;

use ndarray::Array2;
use ndarray_rand::RandomExt;

use itertools::izip;

const N: usize = 1500;

fn initialize_spin_lattice(dim: (usize, usize)) -> Array2<AtomicI8> {
    Array2::random(dim, Uniform::new(0., 1.))
        .mapv(|a| if a > 0.5 { 1 } else { -1 })
        .mapv(|a| AtomicI8::new(a))
}

fn metropolis_step(lattice: &mut Arc<Array2<AtomicI8>>, beta: f64) {
    let n = lattice.shape()[0];
    let m = lattice.shape()[1];

    let x_values = repeat_with(|| fastrand::Rng::new().usize(0..n));
    let y_values = repeat_with(|| fastrand::Rng::new().usize(0..m));
    let r_values = repeat_with(|| fastrand::Rng::new().f64());

    for (x, y, r) in izip!(x_values, y_values, r_values).take(m * n) {
        let ordering = Ordering::Relaxed;
        let right = lattice[[(x + 1) % n, y]].load(ordering);
        let down = lattice[[x, (y + 1) % m]].load(ordering);
        let left = lattice[[x.saturating_sub(1), y]].load(ordering);
        let up = lattice[[x, y.saturating_sub(1)]].load(ordering);
        let curr = &lattice[[x, y]];
        let neighbors_sum = right + down + left + up;
        let delta_e = (2 * curr.load(ordering) * neighbors_sum) as f64;
        if delta_e < 0_f64 || r < (-beta * delta_e).exp() {
            let _ = lattice[[x, y]].fetch_update(ordering, ordering, |v| Some(-v));
        }
    }
}

fn main() {
    nannou::app(model).update(update).simple_window(view).run();
}

struct Model {
    pool: ThreadPool,
    lattice: Arc<Array2<AtomicI8>>,
}

fn model(_app: &App) -> Model {
    Model {
        pool: ThreadPool::new(std::thread::available_parallelism().unwrap().get()),
        lattice: Arc::new(initialize_spin_lattice((N, N))),
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    if _model.pool.queued_count() > 10 {
        return;
    }
    let beta = (_app.mouse.x as f64 + ((N / 2) as f64)) / ((N / 2) as f64);
    let mut lattice_ptr = Arc::clone(&_model.lattice);
    _model
        .pool
        .execute(move || metropolis_step(&mut lattice_ptr, beta));
}

fn view(_app: &App, _model: &Model, frame: Frame) {
    let draw = _app.draw();
    let vec: Vec<u8> = _model
        .lattice
        .iter()
        .map(|x| u8::try_from(255 * (x.load(Ordering::Relaxed) as i16 + 1) / 2).unwrap())
        .collect();
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
