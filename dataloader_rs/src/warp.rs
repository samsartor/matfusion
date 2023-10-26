use anyhow::{bail, Error};
use cgmath::{InnerSpace, Matrix2, Matrix3, Point2, SquareMatrix, Vector2, Vector3};
use ndarray::{s, ArrayView3, ArrayViewMut1, ArrayViewMut3};
#[cfg(feature = "ndarray-linalg")]
use ndarray_linalg::svd::SVD;

#[derive(Debug, Clone, Copy)]
pub struct Query {
    /// The point in the image/SVBRDF we want to sample.
    /// For an SVBRDF this is usually in the range (0-255, 0-255).
    pub point: Point2<f32>,
    /// A matrix that transforms small pertubations in the output pixel location
    /// into pertubations in the `Query.point` coordinate. This is mainly needed
    /// to correctly rotate/distort normal maps.
    pub frame: Option<Matrix2<f32>>,
}

impl Query {
    /// Distorting a normal map requires rotating the X,Y coordinates of the normal.
    pub fn transform_normal(&self, mut normal: Vector3<f32>) -> Option<Vector3<f32>> {
        let mut mat = self
            .frame
            .expect("can not warp normal maps without a frame");
        if mat.x.magnitude() <= f32::EPSILON {
            return None;
        }
        mat.x /= mat.x.magnitude();
        if mat.y.magnitude() <= f32::EPSILON {
            return None;
        }
        mat.y /= mat.y.magnitude();
        let mat = mat.invert()?;
        normal.y *= -1.0; // OpenGL-style normals
        normal = (mat * normal.truncate()).extend(normal.z);
        normal.y *= -1.0;
        Some(normal)
    }
}

pub trait SourceArray {
    /// Sample this source image/SVBRDF at the given point and write the
    /// resulting channels into `dest`. Returns false if the sampled point is
    /// outside the defined area of this image/SVBRDF.
    fn query(&self, q: Query, dest: ArrayViewMut1<f32>) -> bool;
}

pub struct BilinearInterpolation<'a>(pub ArrayView3<'a, f32>);

impl<'a> SourceArray for BilinearInterpolation<'a> {
    fn query(&self, Query { point, .. }: Query, mut dest: ArrayViewMut1<f32>) -> bool {
        fn to_index(v: f32, min: usize, max: usize) -> Option<usize> {
            if v >= min as f32 && v < max as f32 {
                Some(v as usize)
            } else {
                None
            }
        }

        let Some(y0) = to_index(point.y.floor(), 0, self.0.dim().0) else {
            return false;
        };
        let Some(y1) = to_index(point.y.ceil(), 0, self.0.dim().0) else {
            return false;
        };
        let Some(x0) = to_index(point.x.floor(), 0, self.0.dim().1) else {
            return false;
        };
        let Some(x1) = to_index(point.x.ceil(), 0, self.0.dim().1) else {
            return false;
        };
        let v00 = self.0.slice(s![y0, x0, ..]);
        let v01 = self.0.slice(s![y0, x1, ..]);
        let v10 = self.0.slice(s![y1, x0, ..]);
        let v11 = self.0.slice(s![y1, x1, ..]);
        let ty = point.y.fract();
        let sy = 1.0 - ty;
        let tx = point.x.fract();
        let sx = 1.0 - tx;
        for k in 0..dest.len() {
            dest[k] = (v00[k] * sy + v10[k] * ty) * sx + (v01[k] * sy + v11[k] * ty) * tx;
        }
        true
    }
}

pub struct SvbrdfInterpolation<S>(pub S);

impl<S: SourceArray> SourceArray for SvbrdfInterpolation<S> {
    fn query(&self, q: Query, mut dest: ArrayViewMut1<f32>) -> bool {
        if !self.0.query(q, dest.view_mut()) {
            return false;
        }
        let mut n = Vector3::new(dest[7], dest[8], dest[9]);
        n = n * 2.0 - Vector3::new(1.0, 1.0, 1.0);
        let Some(new_n) = q.transform_normal(n) else {
            return false;
        };
        n = new_n;
        n = n.normalize();
        n = n / 2.0 + Vector3::new(0.5, 0.5, 0.5);
        dest[7] = n.x;
        dest[8] = n.y;
        dest[9] = n.z;
        true
    }
}

/// Performs an arbitrary warping of the array `source`, saving the result into
/// `dest`. The `warp` function takes an output coordinate and specifies the
/// corisponding point in `source`, and `fill` provides a background pixel value
/// when `warp` outputs a cordinate outside the defined area of `source`.
pub fn reverse_warp(
    source: impl SourceArray,
    mut dest: ArrayViewMut3<f32>,
    mut warp: impl FnMut(Point2<usize>) -> Option<Query>,
    mut fill: impl FnMut(Point2<usize>, ArrayViewMut1<f32>),
) {
    for i in 0..dest.dim().0 {
        for j in 0..dest.dim().1 {
            let p = Point2 { x: j, y: i };
            let mut dest = dest.slice_mut(s![i, j, ..]);
            'load: {
                let Some(q) = warp(p) else {
                    fill(p, dest);
                    break 'load;
                };
                if !source.query(q, dest.view_mut()) {
                    fill(p, dest);
                    break 'load;
                }
            }
        }
    }
}

/// Downscales the input `source` into `dest` with mean-pooling.
pub fn downcale(source: ArrayView3<f32>, mut dest: ArrayViewMut3<f32>) {
    let (h0, w0, c) = source.dim();
    let (h1, w1, _) = dest.dim();
    for i1 in 0..h1 {
        for j1 in 0..w1 {
            let from_i0 = i1 * h0 / h1;
            let to_i0 = ((i1 + 1) * h0 / h1).max(from_i0 + 1);
            let from_j0 = j1 * w0 / w1;
            let to_j0 = ((j1 + 1) * w0 / w1).max(from_j0 + 1);

            for k in 0..c {
                let mut x = 0.0;
                let mut n = 0;
                for i0 in from_i0..to_i0 {
                    for j0 in from_j0..to_j0 {
                        x += source[(i0, j0, k)];
                        n += 1;
                    }
                }
                dest[(i1, j1, k)] = x / n as f32;
            }
        }
    }
}

/// Resizes `source` into `dest` with bilinear interpolation or mean-pooling as needed.
pub fn resize(source: ArrayView3<f32>, dest: ArrayViewMut3<f32>) {
    let (h0, w0, _) = source.dim();
    let (h1, w1, _) = dest.dim();
    if h0 > h1 * 2 || w0 > w1 * 2 {
        downcale(source, dest);
    } else {
        reverse_warp(
            BilinearInterpolation(source),
            dest,
            |i| {
                Some(Query {
                    point: Point2 {
                        x: i.x as f32 / h1 as f32 * h0 as f32,
                        y: i.y as f32 / w1 as f32 * w0 as f32,
                    },
                    frame: Some(Matrix2::new(1.0, 0.0, 0.0, 1.0)),
                })
            },
            |_, _| (),
        );
    }
}

/// Resizes a cropped area of `source` into `dest`, always using bilinear
/// interpolation.
pub fn resize_and_crop(
    source: ArrayView3<f32>,
    dest: ArrayViewMut3<f32>,
    crop: f32,
    center: Point2<f32>,
) {
    let (h0, w0, _) = source.dim();
    let (h1, w1, _) = dest.dim();
    reverse_warp(
        BilinearInterpolation(source),
        dest,
        |i| {
            let mut point = Point2 {
                x: i.x as f32,
                y: i.y as f32,
            };
            point.x /= h1 as f32;
            point.y /= w1 as f32;
            point = center + (point - center) * crop;
            point.x *= h0 as f32;
            point.y *= w0 as f32;
            Some(Query {
                point,
                frame: Some(Matrix2::new(1.0, 0.0, 0.0, 1.0)),
            })
        },
        |_, _| (),
    );
}

/// A UV map.
pub struct UvBuffer<'a>(pub ArrayView3<'a, f32>);

impl<'a> UvBuffer<'a> {
    /// Given a point in the UV buffer, create a query that would sample the corrisponding texture.
    pub fn query(&self, Point2 { x, y }: Point2<usize>) -> Option<Query> {
        let u = self.0.get([y, x, 0]).copied()?;
        let v = self.0.get([y, x, 1]).copied()?;
        if !u.is_finite() || !v.is_finite() {
            return None;
        }

        // The tricky part here is seeing how the UV coordinate of neiboring pixels differs,
        // and determining how to rotate the normals in any normal map we might sample.

        let x0 = x.saturating_sub(1);
        let x1 = x.saturating_add(1).min(self.0.dim().1 - 1);
        let y0 = y.saturating_sub(1);
        let y1 = y.saturating_add(1).min(self.0.dim().0 - 1);

        let dudy = (*self.0.get([y1, x, 0]).unwrap() - *self.0.get([y0, x, 0]).unwrap()) / 2.0;
        let dudx = (*self.0.get([y, x1, 0]).unwrap() - *self.0.get([y, x0, 0]).unwrap()) / 2.0;
        let dvdy = (*self.0.get([y1, x, 1]).unwrap() - *self.0.get([y0, x, 1]).unwrap()) / 2.0;
        let dvdx = (*self.0.get([y, x1, 1]).unwrap() - *self.0.get([y, x0, 1]).unwrap()) / 2.0;

        let frame = Matrix2 {
            x: Vector2 { x: dudx, y: dvdx },
            y: Vector2 { x: dudy, y: dvdy },
        };

        Some(Query {
            point: Point2 { x: u, y: v },
            frame: Some(frame),
        })
    }
}

/// Given two different arbitrary parameterizations `dest` and `source`, figure
/// out how to smoothly warp arbitrary `source` pixels to the dest pixels that
/// have the closest matching parameter (e.g. UV coordinate).
pub struct NearestBuffer<'a> {
    pub dest: ArrayView3<'a, f32>,
    pub source: UvBuffer<'a>,
    pub nn: kdtree::KdTree<f32, Point2<usize>, &'a [f32]>,
}

impl<'a> NearestBuffer<'a> {
    pub fn new(dest: ArrayView3<'a, f32>, source: ArrayView3<'a, f32>) -> Self {
        let mut nn = kdtree::KdTree::new(source.dim().2);
        for i in 0..source.dim().0 {
            for j in 0..source.dim().1 {
                let _ = nn.add(
                    source.slice_move(s![i, j, ..]).to_slice().unwrap(),
                    Point2 { x: j, y: i },
                );
            }
        }
        let source = UvBuffer(source);
        NearestBuffer { dest, source, nn }
    }

    /// Where should the given point in `self.source` go on `self.dest`, so that
    /// the parameters match up.
    pub fn query(&self, Point2 { x, y }: Point2<usize>, margin: f32) -> Option<Query> {
        let q = self.dest.slice(s![y, x, ..]).to_slice()?;
        let sq_dist =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum() };
        let mut near = self.nn.iter_nearest(q, &sq_dist).ok()?;
        let (d, &pt_i) = near.next()?;
        let mut pt = pt_i.cast::<f32>()?;
        if d > margin.powi(2) {
            // we missed by at least `margin` units, consider there to be no match
            None
        } else {
            let q_uv = Point2 { x: q[0], y: q[1] };
            let Query { point: p_uv, frame } = self.source.query(pt_i)?;
            let err = q_uv - p_uv;
            if let Some(frame) = frame.and_then(|frame| frame.invert()) {
                pt += frame * err;
            };
            Some(Query {
                point: pt,
                // TODO: not currently used, which is good because this would be terrifying to
                // calculate.
                frame: None,
            })
        }
    }

    // Find the homography that best approximates this arbitrary warping.
    pub fn homograpy(&self, ticks: usize, inset: usize, margin: f32) -> Result<Homography, Error> {
        let (w, h, _) = self.dest.dim();
        let mut xs = Vec::with_capacity(ticks * ticks);
        let mut ys = Vec::with_capacity(ticks * ticks);
        for i in 0..ticks {
            let i = (i + inset) * h / (ticks - 1 + inset * 2);
            for j in 0..ticks {
                let j = (j + inset) * w / (ticks - 1 + inset * 2);
                let x = Point2::new(j, i);
                if let Some(y) = self.query(x, margin) {
                    xs.push(x.cast().unwrap());
                    ys.push(y.point);
                }
            }
        }
        solve_homography(&xs, &ys).map(|matrix| Homography { matrix })
    }
}

#[cfg(feature = "ndarray-linalg")]
pub fn solve_homography(x: &[Point2<f32>], y: &[Point2<f32>]) -> Result<Matrix3<f32>, Error> {
    use ndarray::Array2;
    use cgmath::Matrix;

    assert_eq!(x.len(), y.len());
    if x.len() <= 8 {
        bail!("too few points");
    }
    let mut a = Array2::zeros((x.len() * 2, 9));
    for (i, (x, y)) in x.iter().zip(y.iter()).enumerate() {
        let (i, j) = (2 * i, 2 * i + 1);
        a[(i, 0)] = -x.x;
        a[(i, 1)] = -x.y;
        a[(i, 2)] = -1.0;
        a[(j, 3)] = -x.x;
        a[(j, 4)] = -x.y;
        a[(j, 5)] = -1.0;

        a[(i, 6)] = y.x * x.x;
        a[(j, 6)] = y.y * x.x;
        a[(i, 7)] = y.x * x.y;
        a[(j, 7)] = y.y * x.y;
        a[(i, 8)] = y.x;
        a[(j, 8)] = y.y;
    }
    let (_, _, Some(vt)) = a.svd(false, true)? else {
        panic!("svd did not return vt")
    };
    let h = vt
        .slice(s![vt.dim().0 - 1, ..])
        .to_slice()
        .expect("not in standard order");
    let h: &[f32; 9] = h.try_into()?;
    let h: &Matrix3<f32> = h.into();
    Ok(h.transpose())
}

#[cfg(not(feature = "ndarray-linalg"))]
pub fn solve_homography(_x: &[Point2<f32>], _y: &[Point2<f32>]) -> Result<Matrix3<f32>, Error> {
    bail!("homography solving is only availible with the ndarray-linalg feature")
}

pub struct Homography {
    pub matrix: Matrix3<f32>,
}

impl Homography {
    pub fn query(&self, Point2 { x, y }: Point2<usize>) -> Option<Query> {
        let v = self.matrix * Vector3::new(x as f32, y as f32, 1.0);
        Some(Query {
            point: Point2::new(v.x / v.z, v.y / v.z),
            // TODO: not currently used, because this would also be terrifying to calculate
            frame: None,
        })
    }
}
