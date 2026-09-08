use delaunator::{triangulate, Point};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::NonZeroUsize;

#[derive(Copy, Clone, PartialEq)]
struct HeapEntry {
    dens: f64,
    index: usize,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.dens.partial_cmp(&other.dens) {
            Some(Ordering::Equal) | None => other.index.cmp(&self.index),
            Some(ord) => ord,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct EarlyStopper {
    window: usize,
    min_iters: usize,
    rel_tol: f64,
    abs_tol: f64,
    best_history: Vec<f64>,
    iter: usize,
    last: f64,
    under_relax: bool,
}

impl EarlyStopper {
    fn new(window: usize, min_iters: usize, rel_tol: f64, abs_tol: f64) -> Self {
        Self {
            window,
            min_iters,
            rel_tol,
            abs_tol,
            best_history: vec![f64::INFINITY],
            iter: 0,
            last: f64::INFINITY,
            under_relax: false,
        }
    }

    fn update(&mut self, value: f64) -> bool {
        self.iter += 1;
        if value > self.last {
            self.under_relax = true;
        }
        self.last = value;

        let current_best = value.min(*self.best_history.last().unwrap());
        self.best_history.push(current_best);

        if self.iter < self.min_iters.max(self.window) {
            return false;
        }

        let best_then = self.best_history[self.best_history.len() - 1 - self.window];
        let best_now = *self.best_history.last().unwrap();
        let threshold = self.abs_tol.max(self.rel_tol * best_then.abs());
        if best_then - best_now > threshold {
            return false;
        }
        true
    }
}

fn build_adjacency(n: usize, tri_indices: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let num_triangles = tri_indices.len() / 3;
    let mut raw_edges = Vec::with_capacity(num_triangles * 6);
    for t in 0..num_triangles {
        let u = tri_indices[3 * t];
        let v = tri_indices[3 * t + 1];
        let w = tri_indices[3 * t + 2];

        raw_edges.push((u, v));
        raw_edges.push((v, u));
        raw_edges.push((v, w));
        raw_edges.push((w, v));
        raw_edges.push((w, u));
        raw_edges.push((u, w));
    }

    raw_edges.sort_unstable();
    raw_edges.dedup();

    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(raw_edges.len());
    let mut current_u = 0;
    indptr.push(0);

    for &(u, v) in &raw_edges {
        while current_u < u {
            indptr.push(indices.len());
            current_u += 1;
        }
        indices.push(v);
    }
    while current_u < n {
        indptr.push(indices.len());
        current_u += 1;
    }

    (indptr, indices)
}

fn reassign_bad_bins(
    xy: &[[f64; 2]],
    bin_num: &mut [usize],
) -> Vec<[f64; 2]> {
    let n = xy.len();
    let mut good_bins: Vec<usize> = bin_num.iter().copied().filter(|&b| b > 0).collect();
    good_bins.sort_unstable();
    good_bins.dedup();

    let num_good = good_bins.len();
    if num_good == 0 {
        return Vec::new();
    }

    let mut bin_id_to_idx = std::collections::HashMap::new();
    for (idx, &id) in good_bins.iter().enumerate() {
        bin_id_to_idx.insert(id, idx);
    }

    let mut sum_x = vec![0.0; num_good];
    let mut sum_y = vec![0.0; num_good];
    let mut count = vec![0usize; num_good];

    for i in 0..n {
        let b = bin_num[i];
        if b > 0 {
            let idx = bin_id_to_idx[&b];
            sum_x[idx] += xy[i][0];
            sum_y[idx] += xy[i][1];
            count[idx] += 1;
        }
    }

    let mut good_centroids = Vec::with_capacity(num_good);
    for idx in 0..num_good {
        good_centroids.push([sum_x[idx] / count[idx] as f64, sum_y[idx] / count[idx] as f64]);
    }

    let bad_indices: Vec<usize> = (0..n).filter(|&i| bin_num[i] == 0).collect();
    if !bad_indices.is_empty() {
        // Use 2D KD-Tree for fast bad pixel reassignment
        let tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&good_centroids).unwrap();
        for &bad_idx in &bad_indices {
            let p = xy[bad_idx];
            let best_idx = tree
                .query(&p)
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute()
                .item as usize;
            bin_num[bad_idx] = good_bins[best_idx];
        }

        sum_x.fill(0.0);
        sum_y.fill(0.0);
        count.fill(0);

        for i in 0..n {
            let b = bin_num[i];
            let idx = bin_id_to_idx[&b];
            sum_x[idx] += xy[i][0];
            sum_y[idx] += xy[i][1];
            count[idx] += 1;
        }

        for idx in 0..num_good {
            good_centroids[idx] = [sum_x[idx] / count[idx] as f64, sum_y[idx] / count[idx] as f64];
        }
    }

    good_centroids
}

fn bin_accretion(
    xy: &[[f64; 2]],
    dens: &[f64],
    target_capacity: f64,
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let n = xy.len();
    let mut bin_num = vec![0usize; n];
    let mut bad = vec![true; n];

    let pts: Vec<Point> = xy.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let tri = triangulate(&pts);
    let (indptr, indices) = build_adjacency(n, &tri.triangles);

    let mut heap: BinaryHeap<HeapEntry> = (0..n)
        .map(|i| HeapEntry { dens: dens[i], index: i })
        .collect();

    let q = 0.2;
    let fac = (q + 1.0 / q) / (4.0 * std::f64::consts::PI);

    let mut frontier: Vec<usize> = Vec::with_capacity(64);
    let mut current_bin: Vec<usize> = Vec::with_capacity(128);
    let mut frontier_epoch = vec![0u32; n];
    let mut epoch = 0u32;

    let mut ind = 0usize;

    loop {
        let mut seed = None;
        while let Some(entry) = heap.pop() {
            if bin_num[entry.index] == 0 {
                seed = Some(entry.index);
                break;
            }
        }
        let current_seed = match seed {
            Some(s) => s,
            None => break,
        };

        ind += 1;
        epoch += 1;

        bin_num[current_seed] = ind;
        current_bin.clear();
        current_bin.push(current_seed);

        let mut centroid = xy[current_seed];
        let mut pixel_count = 1usize;
        let mut r2_sum = 0.0;
        let mut capacity = dens[current_seed];

        frontier.clear();
        for &nb in &indices[indptr[current_seed]..indptr[current_seed + 1]] {
            if bin_num[nb] == 0 && frontier_epoch[nb] != epoch {
                frontier_epoch[nb] = epoch;
                frontier.push(nb);
            }
        }

        while capacity < target_capacity {
            if frontier.is_empty() {
                break;
            }

            let mut best_jpix = 0;
            let mut min_d2 = f64::INFINITY;
            for (j, &p) in frontier.iter().enumerate() {
                let dx = xy[p][0] - centroid[0];
                let dy = xy[p][1] - centroid[1];
                let d2 = dx * dx + dy * dy;
                if d2 < min_d2 {
                    min_d2 = d2;
                    best_jpix = j;
                }
            }

            let new_pix = frontier[best_jpix];

            let cand_pixel_count = pixel_count + 1;
            let delta = [xy[new_pix][0] - centroid[0], xy[new_pix][1] - centroid[1]];
            let cand_centroid = [
                centroid[0] + delta[0] / cand_pixel_count as f64,
                centroid[1] + delta[1] / cand_pixel_count as f64,
            ];
            let cand_r2_sum = r2_sum + delta[0] * (xy[new_pix][0] - cand_centroid[0])
                + delta[1] * (xy[new_pix][1] - cand_centroid[1]);

            if cand_r2_sum > fac * (cand_pixel_count as f64 * cand_pixel_count as f64) {
                break;
            }

            let capacity_old = capacity;
            capacity = capacity + dens[new_pix];

            if capacity + capacity_old > 2.0 * target_capacity {
                break;
            }

            pixel_count = cand_pixel_count;
            centroid = cand_centroid;
            r2_sum = cand_r2_sum;
            bin_num[new_pix] = ind;
            current_bin.push(new_pix);

            frontier.swap_remove(best_jpix);
            for &nb in &indices[indptr[new_pix]..indptr[new_pix + 1]] {
                if bin_num[nb] == 0 && frontier_epoch[nb] != epoch {
                    frontier_epoch[nb] = epoch;
                    frontier.push(nb);
                }
            }
        }

        if capacity > 0.8 * target_capacity {
            for &pix in &current_bin {
                bad[pix] = false;
            }
        }
    }

    for i in 0..n {
        if bad[i] {
            bin_num[i] = 0;
        }
    }

    let xybin = reassign_bad_bins(xy, &mut bin_num);
    (xybin, bin_num)
}

fn compute_power_diagram(
    xy: &[[f64; 2]],
    xybin: &[[f64; 2]],
    rbin: &[f64],
) -> Vec<usize> {
    let m = xybin.len();
    let rmax = 1.001 * rbin.iter().fold(0.0f64, |acc, &r: &f64| acc.max(r.abs()));
    let rmax2 = rmax * rmax;

    let mut lifted = Vec::with_capacity(m);
    for j in 0..m {
        let r = rbin[j];
        let z = (rmax2 - r * r).max(0.0).sqrt();
        lifted.push([xybin[j][0], xybin[j][1], z]);
    }

    let tree: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&lifted).unwrap();

    xy.par_iter()
        .map(|p| {
            tree.query(&[p[0], p[1], 0.0])
                .nearest_one::<SquaredEuclidean<f64>>()
                .execute()
                .item as usize
        })
        .collect()
}

fn regularization(
    xy: &[[f64; 2]],
    mut xybin: Vec<[f64; 2]>,
    dens: &[f64],
    target_capacity: f64,
    maxiter: usize,
    verbose: usize,
) -> (Vec<usize>, Vec<[f64; 2]>, Vec<f64>, Vec<f64>, Vec<usize>, usize) {
    let _n = xy.len();
    let m = xybin.len();
    let mut rbin = vec![1.0; m];
    let mut stopper = EarlyStopper::new(20, 30, 0.05, 0.05);
    let damp = 0.5;

    let mut npix = vec![0usize; m];
    let mut capacity = vec![0.0; m];
    let mut it_final = 0;

    let chunk_size = 4096;

    for it in 0..maxiter {
        it_final = it;
        let xybin_old = xybin.clone();
        let rbin_old = rbin.clone();

        // 1. Compute Power Diagram and accumulate bin stats in parallel via fold/reduce
        let rmax = 1.001 * rbin.iter().fold(0.0f64, |acc, &r: &f64| acc.max(r.abs()));
        let rmax2 = rmax * rmax;
        let mut lifted = Vec::with_capacity(m);
        for j in 0..m {
            let r = rbin[j];
            let z = (rmax2 - r * r).max(0.0).sqrt();
            lifted.push([xybin[j][0], xybin[j][1], z]);
        }
        let tree: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&lifted).unwrap();

        let (sum_x, sum_y, counts, cap_acc) = xy.par_chunks(chunk_size)
            .enumerate()
            .fold(
                || (vec![0.0; m], vec![0.0; m], vec![0usize; m], vec![0.0; m]),
                |(mut sx, mut sy, mut cnt, mut cacc), (chunk_idx, chunk)| {
                    let start = chunk_idx * chunk_size;
                    for (offset, &p) in chunk.iter().enumerate() {
                        let i = start + offset;
                        let b = tree.query(&[p[0], p[1], 0.0])
                            .nearest_one::<SquaredEuclidean<f64>>()
                            .execute()
                            .item as usize;
                        cnt[b] += 1;
                        sx[b] += p[0];
                        sy[b] += p[1];
                        cacc[b] += dens[i];
                    }
                    (sx, sy, cnt, cacc)
                }
            )
            .reduce(
                || (vec![0.0; m], vec![0.0; m], vec![0usize; m], vec![0.0; m]),
                |mut a, b| {
                    for j in 0..m {
                        a.0[j] += b.0[j];
                        a.1[j] += b.1[j];
                        a.2[j] += b.2[j];
                        a.3[j] += b.3[j];
                    }
                    a
                }
            );

        npix = counts;
        capacity = cap_acc;

        // Update centroids
        for j in 0..m {
            if npix[j] > 0 {
                xybin[j][0] = sum_x[j] / npix[j] as f64;
                xybin[j][1] = sum_y[j] / npix[j] as f64;
            }
        }

        // Update radii
        for j in 0..m {
            let fac = target_capacity / capacity[j];
            rbin[j] = (fac * npix[j] as f64 / std::f64::consts::PI).sqrt();
        }

        // Under-relaxation
        if stopper.under_relax {
            for j in 0..m {
                xybin[j][0] = xybin_old[j][0] + damp * (xybin[j][0] - xybin_old[j][0]);
                xybin[j][1] = xybin_old[j][1] + damp * (xybin[j][1] - xybin_old[j][1]);
                rbin[j] = rbin_old[j] + damp * (rbin[j] - rbin_old[j]);
            }
        }

        // Nearest neighbors of every bin
        let gen_tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&xybin).unwrap();
        let dists: Vec<f64> = xybin
            .par_iter()
            .map(|pt| {
                let res = gen_tree
                    .query(pt)
                    .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(2).unwrap())
                    .execute();
                if res.len() > 1 {
                    res[1].distance.sqrt()
                } else {
                    1.0
                }
            })
            .collect();

        for j in 0..m {
            let upper = dists[j] - 0.5;
            rbin[j] = rbin[j].clamp(0.5, upper.max(0.5));
        }

        let mut diff2 = 0.0;
        for j in 0..m {
            let dx = xybin[j][0] - xybin_old[j][0];
            let dy = xybin[j][1] - xybin_old[j][1];
            diff2 += dx * dx + dy * dy;
        }
        let diff = diff2.sqrt();

        if verbose >= 2 {
            println!("Iter: {:4}  Diff: {:.3}", it, diff);
        }

        if diff < 0.1 {
            if verbose >= 2 {
                println!("Converged");
            }
            break;
        }
        if stopper.update(diff) {
            if verbose >= 2 {
                println!("Cycling over last {} iterations", stopper.window);
            }
            break;
        }
    }

    // Final Power Diagram assignment
    let bin_num = compute_power_diagram(xy, &xybin, &rbin);

    (bin_num, xybin, rbin, capacity, npix, it_final)
}

fn main() {
    let file = File::open(".venv/lib/python3.14/site-packages/powerbin/examples/sample_data_ngc2273.txt").expect("open file");
    let reader = BufReader::new(file);
    let mut xy = Vec::new();
    let mut dens = Vec::new();

    for line in reader.lines() {
        let line = line.expect("line");
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            let x: f64 = parts[0].parse().unwrap();
            let y: f64 = parts[1].parse().unwrap();
            let sig: f64 = parts[2].parse().unwrap();
            let noise: f64 = parts[3].parse().unwrap();
            xy.push([x, y]);
            dens.push((sig / noise) * (sig / noise));
        }
    }

    let pixelsize = 0.7999080004238518;
    for p in &mut xy {
        p[0] /= pixelsize;
        p[1] /= pixelsize;
    }

    let target_capacity = 2500.0;

    let t0 = std::time::Instant::now();
    let (xybin, _) = bin_accretion(&xy, &dens, target_capacity);
    let t1 = std::time::Instant::now();
    let (_bin_num, _xybin, rbin, bin_capacity, npix, it) = regularization(&xy, xybin, &dens, target_capacity, 50, 2);
    let t2 = std::time::Instant::now();

    println!("Accretion time: {:?}", t1 - t0);
    println!("Regularization time ({} iters): {:?}", it, t2 - t1);
    println!("Total time: {:?}", t2 - t0);

    let non_single_caps: Vec<f64> = (0..npix.len())
        .filter(|&j| npix[j] > 1)
        .map(|j| bin_capacity[j])
        .collect();
    let n_non_single = non_single_caps.len();
    let mean_cap = non_single_caps.iter().sum::<f64>() / n_non_single as f64;
    let var = non_single_caps.iter().map(|&c| (c - mean_cap) * (c - mean_cap)).sum::<f64>() / (n_non_single - 1) as f64;
    let rms_frac = var.sqrt() / mean_cap * 100.0;

    let single_count = npix.iter().filter(|&&c| c == 1).count();
    println!("Bins: {}, Single Pixels: {}/{}", rbin.len(), single_count, xy.len());
    println!("Capacity Fractional RMS Scatter (%): {:.2}", rms_frac);
}
