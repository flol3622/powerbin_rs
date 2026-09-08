use crate::geometry::build_delaunay_adjacency;
use delaunator::{Point, triangulate};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

pub fn reassign_bad_bins(xy: &[[f64; 2]], bin_num: &mut [usize]) -> Vec<[f64; 2]> {
    let n = xy.len();
    let mut good_bins: Vec<usize> = bin_num.iter().copied().filter(|&b| b > 0).collect();
    good_bins.sort_unstable();
    good_bins.dedup();

    let num_good = good_bins.len();
    if num_good == 0 {
        return Vec::new();
    }

    let mut bin_id_to_idx = std::collections::HashMap::with_capacity(num_good);
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
        good_centroids.push([
            sum_x[idx] / count[idx] as f64,
            sum_y[idx] / count[idx] as f64,
        ]);
    }

    let bad_indices: Vec<usize> = (0..n).filter(|&i| bin_num[i] == 0).collect();
    if !bad_indices.is_empty() {
        let tree: ImmutableKdTree<f64, 2> =
            ImmutableKdTree::new_from_slice(&good_centroids).unwrap();
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
            good_centroids[idx] = [
                sum_x[idx] / count[idx] as f64,
                sum_y[idx] / count[idx] as f64,
            ];
        }
    }

    good_centroids
}

use crate::CapacityFn;

pub fn bin_accretion_impl(
    xy: &[[f64; 2]],
    dens: &[f64],
    target_capacity: f64,
    custom_capacity_fn: Option<CapacityFn>,
    verbose: usize,
) -> (Vec<[f64; 2]>, Vec<f64>, Vec<usize>) {
    let n = xy.len();
    if verbose >= 1 {
        println!("Bin-accretion Delaunay...");
    }

    let mut bin_num = vec![0usize; n];
    let mut bad = vec![true; n];

    let pts: Vec<Point> = xy.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let tri = triangulate(&pts);
    let (indptr, indices) = build_delaunay_adjacency(n, &tri.triangles);

    let mut heap: BinaryHeap<HeapEntry> = (0..n)
        .map(|i| HeapEntry {
            dens: dens[i],
            index: i,
        })
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
            let cand_r2_sum = r2_sum
                + delta[0] * (xy[new_pix][0] - cand_centroid[0])
                + delta[1] * (xy[new_pix][1] - cand_centroid[1]);

            if cand_r2_sum > fac * (cand_pixel_count as f64 * cand_pixel_count as f64) {
                break;
            }

            let capacity_old = capacity;
            capacity = if let Some(func) = custom_capacity_fn {
                current_bin.push(new_pix);
                let c = func(&current_bin);
                current_bin.pop();
                c
            } else {
                capacity + dens[new_pix]
            };

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
    if verbose >= 1 {
        println!("{} initial bins.", ind);
        println!("{} good bins.", xybin.len());
    }

    (xybin, dens.to_vec(), bin_num)
}
