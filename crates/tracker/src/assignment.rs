//! Optimal one-to-one assignment with a cost limit.
//!
//! Solves the rectangular linear assignment problem exactly (Hungarian
//! algorithm, O(n³)), where a row/column pair may only be matched if its
//! cost is below `cost_limit`. That limit is enforced *inside* the
//! optimisation, as the reference ByteTrack does with `lap.lapjv(...,
//! extend_cost=True, cost_limit=...)`: the matrix is padded with dummy
//! rows/columns costing `cost_limit / 2`, so leaving a row and a column
//! both unmatched costs exactly `cost_limit`, and any pair at or above the
//! limit is never worth taking. Filtering after an unconstrained
//! assignment would instead let one over-limit pair displace a valid one.

/// Result of [`assign`]: matched `(row, col)` pairs plus leftovers.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Assignment {
    pub matches: Vec<(usize, usize)>,
    pub unmatched_rows: Vec<usize>,
    pub unmatched_cols: Vec<usize>,
}

/// `cost[r][c]` for `rows` × `cols`; pairs with cost ≥ `cost_limit` are
/// never matched.
pub fn assign(cost: &[Vec<f64>], rows: usize, cols: usize, cost_limit: f64) -> Assignment {
    if rows == 0 || cols == 0 {
        return Assignment {
            matches: Vec::new(),
            unmatched_rows: (0..rows).collect(),
            unmatched_cols: (0..cols).collect(),
        };
    }
    let n = rows + cols;
    let dummy = cost_limit / 2.0;
    let ext = |i: usize, j: usize| -> f64 {
        match (i < rows, j < cols) {
            (true, true) => cost[i][j],
            (false, false) => 0.0,
            _ => dummy,
        }
    };
    let row_to_col = hungarian(n, ext);

    let mut out = Assignment::default();
    let mut col_used = vec![false; cols];
    for (r, &c) in row_to_col.iter().enumerate().take(rows) {
        if c < cols && cost[r][c] < cost_limit {
            out.matches.push((r, c));
            col_used[c] = true;
        } else {
            out.unmatched_rows.push(r);
        }
    }
    out.unmatched_cols = (0..cols).filter(|&c| !col_used[c]).collect();
    out
}

/// Minimum-cost perfect matching on an `n`×`n` matrix given by `cost`.
/// Returns, for each row, its assigned column. Shortest-augmenting-path
/// formulation with row/column potentials.
fn hungarian(n: usize, cost: impl Fn(usize, usize) -> f64) -> Vec<usize> {
    const INF: f64 = f64::INFINITY;
    // 1-indexed internally; index 0 is the virtual start column.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1]; // p[col] = row matched to col
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![INF; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut row_to_col = vec![0; n];
    for j in 1..=n {
        if p[j] > 0 {
            row_to_col[p[j] - 1] = j - 1;
        }
    }
    row_to_col
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: &[&[f64]]) -> Vec<Vec<f64>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn finds_the_global_optimum_not_the_greedy_one() {
        // Greedy (take 0.1 first) would force row 1 onto 0.9 = total 1.0;
        // the optimum is 0.2 + 0.3 = 0.5.
        let cost = m(&[&[0.1, 0.2], &[0.3, 0.9]]);
        let a = assign(&cost, 2, 2, 1.0);
        let mut matches = a.matches.clone();
        matches.sort();
        assert_eq!(matches, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn respects_the_cost_limit_inside_the_optimisation() {
        // Only (0,0)=0.5 and (1,0)=0.0 are under the 0.8 limit, and both
        // want column 0. Taking (1,0) and leaving row 0 / col 1 unmatched
        // costs 0.0 + 0.8; taking (0,0) costs 0.5 + 0.8. An unconstrained
        // solve would happily pick (0,1)=0.85 + (1,0)=0.0 and only then
        // discard 0.85 — the right answer is exactly (1,0).
        let cost = m(&[&[0.5, 0.85], &[0.0, 0.95]]);
        let a = assign(&cost, 2, 2, 0.8);
        assert_eq!(a.matches, vec![(1, 0)]);
        assert_eq!(a.unmatched_rows, vec![0]);
        assert_eq!(a.unmatched_cols, vec![1]);
    }

    #[test]
    fn handles_rectangular_and_empty_inputs() {
        let cost = m(&[&[0.9, 0.1, 0.5]]);
        let a = assign(&cost, 1, 3, 0.8);
        assert_eq!(a.matches, vec![(0, 1)]);
        assert_eq!(a.unmatched_cols, vec![0, 2]);

        let a = assign(&[], 0, 4, 0.8);
        assert!(a.matches.is_empty());
        assert_eq!(a.unmatched_cols, vec![0, 1, 2, 3]);
    }

    #[test]
    fn all_over_limit_matches_nothing() {
        let cost = m(&[&[0.9, 0.95], &[0.99, 1.0]]);
        let a = assign(&cost, 2, 2, 0.8);
        assert!(a.matches.is_empty());
        assert_eq!(a.unmatched_rows, vec![0, 1]);
        assert_eq!(a.unmatched_cols, vec![0, 1]);
    }
}
