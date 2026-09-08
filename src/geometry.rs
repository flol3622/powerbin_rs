use delaunator::{triangulate, Point};

/// Builds CSR representation of Delaunay triangulation adjacency graph
/// with an ultra-fast O(N) bucket-based insertion and local deduplication.
pub fn build_delaunay_adjacency(n: usize, tri_indices: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let num_triangles = tri_indices.len() / 3;
    let mut counts = vec![0usize; n];
    for t in 0..num_triangles {
        let u = tri_indices[3 * t];
        let v = tri_indices[3 * t + 1];
        let w = tri_indices[3 * t + 2];
        counts[u] += 2;
        counts[v] += 2;
        counts[w] += 2;
    }

    let mut indptr = Vec::with_capacity(n + 1);
    let mut cur = 0;
    for &c in &counts {
        indptr.push(cur);
        cur += c;
    }
    indptr.push(cur);

    let mut indices = vec![0usize; cur];
    let mut head = indptr.clone();

    for t in 0..num_triangles {
        let u = tri_indices[3 * t];
        let v = tri_indices[3 * t + 1];
        let w = tri_indices[3 * t + 2];

        indices[head[u]] = v;
        head[u] += 1;
        indices[head[u]] = w;
        head[u] += 1;

        indices[head[v]] = u;
        head[v] += 1;
        indices[head[v]] = w;
        head[v] += 1;

        indices[head[w]] = u;
        head[w] += 1;
        indices[head[w]] = v;
        head[w] += 1;
    }

    let mut final_indptr = Vec::with_capacity(n + 1);
    let mut final_indices = Vec::with_capacity(cur / 2);
    final_indptr.push(0);

    for i in 0..n {
        let start = indptr[i];
        let end = head[i];
        let slice = &mut indices[start..end];
        slice.sort_unstable();
        let mut prev = usize::MAX;
        for &nb in slice.iter() {
            if nb != prev {
                final_indices.push(nb);
                prev = nb;
            }
        }
        final_indptr.push(final_indices.len());
    }

    (final_indptr, final_indices)
}

/// Estimates pixel size from coordinates by taking the median distance
/// to each pixel's nearest neighbor.
pub fn estimate_pixelsize(xy: &[[f64; 2]]) -> f64 {
    let n = xy.len();
    if n < 2 {
        return 1.0;
    }

    let pts: Vec<Point> = xy.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let tri = triangulate(&pts);
    let (indptr, indices) = build_delaunay_adjacency(n, &tri.triangles);

    let mut min_dists = Vec::with_capacity(n);
    for i in 0..n {
        let start = indptr[i];
        let end = indptr[i + 1];
        let mut min_d2 = f64::INFINITY;
        let p = xy[i];
        for &nb in &indices[start..end] {
            let dx = xy[nb][0] - p[0];
            let dy = xy[nb][1] - p[1];
            let d2 = dx * dx + dy * dy;
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }
        if min_d2.is_finite() {
            min_dists.push(min_d2.sqrt());
        }
    }

    if min_dists.is_empty() {
        return 1.0;
    }

    let mid = min_dists.len() / 2;
    min_dists.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if min_dists.len() % 2 == 1 {
        min_dists[mid]
    } else {
        let val1 = min_dists[mid];
        let val0 = *min_dists[..mid].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        0.5 * (val0 + val1)
    }
}
