use crate::stopper::EarlyStopper;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;
use std::num::NonZeroUsize;

/// Computes the Power Diagram assignment for every pixel in `xy`.
pub fn power_diagram(xy: &[[f64; 2]], xybin: &[[f64; 2]], rbin: &[f64]) -> Vec<usize> {
    let m = xybin.len();
    if m == 0 {
        return vec![0; xy.len()];
    }

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

use crate::CapacityFn;

/// Return type of the regularization step:
/// (bin_num, xybin, rbin, bin_capacity, npix, final_iter)
pub type RegularizationOutput = (
    Vec<usize>,
    Vec<[f64; 2]>,
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    usize,
);

/// Updates bin properties based on Power Diagram tessellation.
pub fn update_bins(
    xy: &[[f64; 2]],
    xybin: &[[f64; 2]],
    rbin: &[f64],
    dens: &[f64],
    custom_capacity_fn: Option<CapacityFn>,
) -> (Vec<[f64; 2]>, Vec<usize>, Vec<f64>, Vec<usize>) {
    let n = xy.len();
    let m = xybin.len();
    let bin_num = power_diagram(xy, xybin, rbin);

    let mut npix = vec![0usize; m];
    let mut sum_x = vec![0.0; m];
    let mut sum_y = vec![0.0; m];
    let mut capacity = vec![0.0; m];

    for (i, &b) in bin_num.iter().enumerate().take(n) {
        if b < m {
            npix[b] += 1;
            sum_x[b] += xy[i][0];
            sum_y[b] += xy[i][1];
            capacity[b] += dens[i];
        }
    }

    let mut new_xybin = xybin.to_vec();
    for j in 0..m {
        if npix[j] > 0 {
            new_xybin[j][0] = sum_x[j] / npix[j] as f64;
            new_xybin[j][1] = sum_y[j] / npix[j] as f64;
        }
    }

    if let Some(func) = custom_capacity_fn {
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (i, &b) in bin_num.iter().enumerate().take(n) {
            if b < m {
                groups[b].push(i);
            }
        }
        for j in 0..m {
            capacity[j] = func(&groups[j]);
        }
    }

    (new_xybin, npix, capacity, bin_num)
}

pub fn regularization_impl(
    xy: &[[f64; 2]],
    mut xybin: Vec<[f64; 2]>,
    dens: &[f64],
    target_capacity: f64,
    custom_capacity_fn: Option<CapacityFn>,
    verbose: usize,
    maxiter: usize,
) -> RegularizationOutput {
    let n = xy.len();
    let m = xybin.len();
    let mut rbin = vec![1.0; m];
    let mut stopper = EarlyStopper::new(20, 30, 0.05, 0.05);
    let damp = 0.5;
    let mut under_relax_printed = false;

    let mut npix = vec![0usize; m];
    let mut capacity = vec![0.0; m];
    let mut it_final = 0;

    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n / (num_threads * 4)).max(1024);

    let mut lifted = vec![[0.0f64; 3]; m];

    for it in 0..maxiter {
        it_final = it;
        let xybin_old = xybin.clone();
        let rbin_old = rbin.clone();

        // 1. Lift generators to 3D and query nearest neighbor
        let rmax = 1.001 * rbin.iter().fold(0.0f64, |acc, &r: &f64| acc.max(r.abs()));
        let rmax2 = rmax * rmax;
        for j in 0..m {
            let r = rbin[j];
            let z = (rmax2 - r * r).max(0.0).sqrt();
            lifted[j] = [xybin[j][0], xybin[j][1], z];
        }
        let tree: ImmutableKdTree<f64, 3> =
            ImmutableKdTree::new_from_slice_parallel(&lifted).unwrap();

        if let Some(func) = custom_capacity_fn {
            // Callable capacity: we need the full pixel assignment
            let bin_num: Vec<usize> = xy
                .par_iter()
                .map(|p| {
                    tree.query(&[p[0], p[1], 0.0])
                        .nearest_one::<SquaredEuclidean<f64>>()
                        .execute()
                        .item as usize
                })
                .collect();

            let mut sum_x = vec![0.0; m];
            let mut sum_y = vec![0.0; m];
            npix.fill(0);
            let mut groups: Vec<Vec<usize>> = vec![Vec::new(); m];

            for i in 0..n {
                let b = bin_num[i];
                if b < m {
                    npix[b] += 1;
                    sum_x[b] += xy[i][0];
                    sum_y[b] += xy[i][1];
                    groups[b].push(i);
                }
            }

            for j in 0..m {
                if npix[j] > 0 {
                    xybin[j][0] = sum_x[j] / npix[j] as f64;
                    xybin[j][1] = sum_y[j] / npix[j] as f64;
                }
                capacity[j] = func(&groups[j]);
            }
        } else {
            // Additive capacity: parallel fold/reduce without extra memory allocation
            let (sum_x, sum_y, counts, cap_acc) = xy
                .par_chunks(chunk_size)
                .enumerate()
                .fold(
                    || (vec![0.0; m], vec![0.0; m], vec![0usize; m], vec![0.0; m]),
                    |(mut sx, mut sy, mut cnt, mut cacc), (chunk_idx, chunk)| {
                        let start = chunk_idx * chunk_size;
                        for (offset, &p) in chunk.iter().enumerate() {
                            let i = start + offset;
                            let b = tree
                                .query(&[p[0], p[1], 0.0])
                                .nearest_one::<SquaredEuclidean<f64>>()
                                .execute()
                                .item as usize;
                            if b < m {
                                cnt[b] += 1;
                                sx[b] += p[0];
                                sy[b] += p[1];
                                cacc[b] += dens[i];
                            }
                        }
                        (sx, sy, cnt, cacc)
                    },
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
                    },
                );

            npix = counts;
            capacity = cap_acc;

            for j in 0..m {
                if npix[j] > 0 {
                    xybin[j][0] = sum_x[j] / npix[j] as f64;
                    xybin[j][1] = sum_y[j] / npix[j] as f64;
                }
            }
        }

        // Update radii
        for j in 0..m {
            let fac = target_capacity / capacity[j];
            rbin[j] = (fac * npix[j] as f64 / std::f64::consts::PI).sqrt();
        }

        // Under-relaxation step to reduce cycling
        if stopper.under_relax {
            if verbose >= 2 && !under_relax_printed {
                println!("Under-relaxation started");
                under_relax_printed = true;
            }
            for j in 0..m {
                xybin[j][0] = xybin_old[j][0] + damp * (xybin[j][0] - xybin_old[j][0]);
                xybin[j][1] = xybin_old[j][1] + damp * (xybin[j][1] - xybin_old[j][1]);
                rbin[j] = rbin_old[j] + damp * (rbin[j] - rbin_old[j]);
            }
        }

        // Nearest neighbours of every bin
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
        if it >= maxiter - 1 && verbose >= 2 {
            println!("Reached maximum number of iterations");
        }
    }

    // Final Power Diagram assignment
    let bin_num = power_diagram(xy, &xybin, &rbin);

    (bin_num, xybin, rbin, capacity, npix, it_final)
}
