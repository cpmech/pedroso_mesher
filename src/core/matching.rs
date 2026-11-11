//! Clean-room Edmonds Blossom algorithm for maximum-weight and
//! maximum-cardinality maximum-weight matching in general undirected graphs.
//!
//! Implemented strictly from public mathematical literature:
//! - Edmonds, J. (1965). "Paths, trees, and flowers." *Canadian Journal of
//!   Mathematics*, 17: 449–467.
//! - Edmonds, J. (1965). "Maximum matching and a polyhedron with 0,1-vertices."
//!   *Journal of Research of the National Bureau of Standards*, 69B: 125–130.
//! - Galil, Z. (1986). "Efficient algorithms for finding maximum matching in
//!   graphs." *ACM Computing Surveys*, 18(1): 23–38.
//!
//! All dual variables (`y` on vertices, `z` on odd-set blossoms) are scaled
//! exact integers (`i128`), avoiding floating-point drift and overflow.

/// Sentinel index indicating an unmatched vertex or absent edge/blossom.
pub const UNMATCHED: usize = usize::MAX;

/// Compatibility alias matching the former `SENTINEL` constant.
pub const SENTINEL: usize = UNMATCHED;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    Unlabeled,
    Outer,
    Inner,
}

#[derive(Clone, Copy, Debug)]
struct GraphEdge {
    u: usize,
    v: usize,
    weight: i128,
}

#[derive(Clone, Debug)]
struct BlossomNode {
    parent: usize,
    base: usize,
    label: Label,
    tree_edge: usize,
    dual_z: i128,
    /// Immediate sub-blossoms around the odd cycle `[c_0, c_1, ..., c_{2k}]`,
    /// where `c_0` contains `base`. Empty for trivial blossoms (`0..n`).
    children: Vec<usize>,
    /// Oriented cycle edges `(edge_idx, u_in_ci, v_in_next)` connecting
    /// `children[i]` to `children[(i + 1) % L]`.
    cycle_edges: Vec<(usize, usize, usize)>,
}

impl BlossomNode {
    fn trivial(v: usize) -> Self {
        Self {
            parent: UNMATCHED,
            base: v,
            label: Label::Unlabeled,
            tree_edge: UNMATCHED,
            dual_z: 0,
            children: Vec::new(),
            cycle_edges: Vec::new(),
        }
    }
}

/// Builder-style solver for general-graph weighted matching.
#[derive(Clone, Debug)]
pub struct Matching {
    edges: Vec<(usize, usize, i64)>,
    max_card: bool,
}

impl Matching {
    /// Construct a matching problem from weighted undirected edges `(u, v, w)`.
    pub fn new<W: Into<i64> + Copy>(edges: Vec<(usize, usize, W)>) -> Self {
        let converted = edges
            .into_iter()
            .map(|(u, v, w)| (u, v, w.into()))
            .collect();
        Self {
            edges: converted,
            max_card: false,
        }
    }

    /// Require maximum-cardinality matching first, breaking ties by maximum
    /// total weight.
    pub fn max_cardinality(&mut self) -> &mut Self {
        self.max_card = true;
        self
    }

    /// Compute the optimal `mate` vector (`mate[u] = v` or [`UNMATCHED`]).
    pub fn solve(&mut self) -> Vec<usize> {
        let n = self
            .edges
            .iter()
            .fold(0usize, |acc, &(u, v, _)| acc.max(u + 1).max(v + 1));
        max_weight_matching(n, &self.edges, self.max_card)
    }
}

/// Compute a maximum-weight (or maximum-cardinality, maximum-weight) matching
/// on `n_vertices` (`0..n_vertices`).
///
/// Returns `mate` of length `n_vertices`, where `mate[u] == v` and
/// `mate[v] == u` for matched pairs, and `mate[w] == UNMATCHED` otherwise.
pub fn max_weight_matching(
    n_vertices: usize,
    raw_edges: &[(usize, usize, i64)],
    max_cardinality: bool,
) -> Vec<usize> {
    if n_vertices < 2 || raw_edges.is_empty() {
        return vec![UNMATCHED; n_vertices];
    }

    // Deduplicate parallel edges (keep maximum weight) and drop self-loops /
    // out-of-range endpoints.
    let mut dedup: std::collections::BTreeMap<(usize, usize), i64> =
        std::collections::BTreeMap::new();
    for &(a, b, w) in raw_edges {
        if a == b || a >= n_vertices || b >= n_vertices {
            continue;
        }
        if !max_cardinality && w <= 0 {
            continue;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        match dedup.get_mut(&key) {
            Some(prev) => {
                if w > *prev {
                    *prev = w;
                }
            }
            None => {
                dedup.insert(key, w);
            }
        }
    }
    if dedup.is_empty() {
        return vec![UNMATCHED; n_vertices];
    }

    let min_w = dedup.values().copied().min().unwrap_or(0) as i128;
    let max_w = dedup.values().copied().max().unwrap_or(0) as i128;
    let shift = if max_cardinality {
        let span = (max_w - min_w) + 1;
        (n_vertices as i128 + 1) * span + 1 - min_w + 1
    } else {
        0
    };

    // Multiply shifted positive weights by 2 so all vertex duals `y` and
    // blossom duals `z` stay exact integers throughout primal-dual steps.
    let mut edges = Vec::with_capacity(dedup.len());
    let mut adj = vec![Vec::new(); n_vertices];
    let mut max_half_weight = 0i128;
    for (&(u, v), &w) in &dedup {
        let shifted = (w as i128) + shift;
        if shifted <= 0 {
            continue;
        }
        if shifted > max_half_weight {
            max_half_weight = shifted;
        }
        let eid = edges.len();
        edges.push(GraphEdge {
            u,
            v,
            weight: 2 * shifted,
        });
        adj[u].push(eid);
        adj[v].push(eid);
    }
    if edges.is_empty() {
        return vec![UNMATCHED; n_vertices];
    }

    let mut solver = EdmondsSolver::new(n_vertices, edges, adj, max_half_weight);
    solver.solve()
}

struct EdmondsSolver {
    n: usize,
    edges: Vec<GraphEdge>,
    adj: Vec<Vec<usize>>,
    dual_y: Vec<i128>,
    mate_v: Vec<usize>,
    mate_e: Vec<usize>,
    in_blossom: Vec<usize>,
    blossoms: Vec<BlossomNode>,
    free_blossoms: Vec<usize>,
    queue: Vec<usize>,
    /// Per-blossom best tight/slack edge tracker Scratch for O(n^2 + m) stages.
    best_edge_to: Vec<usize>,
}

impl EdmondsSolver {
    fn new(n: usize, edges: Vec<GraphEdge>, adj: Vec<Vec<usize>>, initial_y: i128) -> Self {
        let total_blossoms = 2 * n;
        let mut blossoms = Vec::with_capacity(total_blossoms);
        for v in 0..n {
            blossoms.push(BlossomNode::trivial(v));
        }
        for _ in n..total_blossoms {
            blossoms.push(BlossomNode::trivial(0));
        }
        let free_blossoms: Vec<usize> = (n..total_blossoms).rev().collect();
        Self {
            n,
            edges,
            adj,
            dual_y: vec![initial_y; n],
            mate_v: vec![UNMATCHED; n],
            mate_e: vec![UNMATCHED; n],
            in_blossom: (0..n).collect(),
            blossoms,
            free_blossoms,
            queue: Vec::with_capacity(n),
            best_edge_to: vec![UNMATCHED; total_blossoms],
        }
    }

    #[inline]
    fn slack(&self, eid: usize) -> i128 {
        let e = self.edges[eid];
        self.dual_y[e.u] + self.dual_y[e.v] - e.weight
    }

    #[inline]
    fn other_endpoint(&self, eid: usize, u: usize) -> usize {
        let e = self.edges[eid];
        if e.u == u {
            e.v
        } else {
            e.u
        }
    }

    /// Collect all graph vertices (`0..n`) inside blossom `b` in deterministic order.
    fn blossom_leaves(&self, b: usize, out: &mut Vec<usize>) {
        out.clear();
        if b < self.n {
            out.push(b);
            return;
        }
        let mut stack = vec![b];
        while let Some(curr) = stack.pop() {
            if curr < self.n {
                out.push(curr);
            } else {
                for &child in self.blossoms[curr].children.iter().rev() {
                    stack.push(child);
                }
            }
        }
    }

    /// Label top-level blossom `b` as `Outer` and enqueue all its vertices.
    fn assign_outer(&mut self, b: usize, tree_edge: usize) {
        self.blossoms[b].label = Label::Outer;
        self.blossoms[b].tree_edge = tree_edge;
        let mut leaves = Vec::new();
        self.blossom_leaves(b, &mut leaves);
        self.queue.extend(leaves);
    }

    /// Label top-level blossom `b` as `Inner`, and label its matched partner
    /// blossom as `Outer`.
    fn assign_inner(&mut self, b: usize, tree_edge: usize) {
        self.blossoms[b].label = Label::Inner;
        self.blossoms[b].tree_edge = tree_edge;
        let base = self.blossoms[b].base;
        let mate_vertex = self.mate_v[base];
        let mate_edge = self.mate_e[base];
        if mate_vertex != UNMATCHED {
            let mate_blossom = self.in_blossom[mate_vertex];
            self.assign_outer(mate_blossom, mate_edge);
        }
    }

    /// Step one hop toward the alternating-tree root from top-level blossom `b`.
    /// Returns `Some((parent_blossom, edge_idx, vertex_in_b, vertex_in_parent))`.
    fn tree_step_up(&self, b: usize) -> Option<(usize, usize, usize, usize)> {
        let te = self.blossoms[b].tree_edge;
        if te == UNMATCHED {
            return None;
        }
        let e = self.edges[te];
        if self.in_blossom[e.u] == b {
            Some((self.in_blossom[e.v], te, e.u, e.v))
        } else {
            Some((self.in_blossom[e.u], te, e.v, e.u))
        }
    }

    /// Find Lowest Common Ancestor of two `Outer` top-level blossoms `bu`, `bv`
    /// if they belong to the same alternating tree; `None` if they have distinct roots.
    fn find_lca(&self, bu: usize, bv: usize) -> Option<usize> {
        let mut visited = vec![false; self.blossoms.len()];
        let mut a = bu;
        let mut b = bv;
        loop {
            if a != UNMATCHED {
                if visited[a] {
                    return Some(a);
                }
                visited[a] = true;
                a = self.tree_step_up(a).map_or(UNMATCHED, |(p, _, _, _)| p);
            }
            if b != UNMATCHED {
                if visited[b] {
                    return Some(b);
                }
                visited[b] = true;
                b = self.tree_step_up(b).map_or(UNMATCHED, |(p, _, _, _)| p);
            }
            if a == UNMATCHED && b == UNMATCHED {
                return None;
            }
        }
    }

    /// Shrink an odd cycle formed by tight edge `eid` between `u` (in `bu`) and
    /// `v` (in `bv`) with common `Outer` ancestor `lca`.
    fn add_blossom(&mut self, eid: usize, u: usize, v: usize, lca: usize) {
        let bu = self.in_blossom[u];
        let bv = self.in_blossom[v];
        let Some(new_b) = self.free_blossoms.pop() else {
            return;
        };

        // Trace path 1: bu -> ... -> lca
        let mut path_u: Vec<usize> = Vec::new();
        let mut edges_u: Vec<(usize, usize, usize)> = Vec::new(); // (eid, v_in_parent, v_in_curr)
        let mut curr = bu;
        while curr != lca {
            path_u.push(curr);
            let Some((parent, te, v_curr, v_parent)) = self.tree_step_up(curr) else {
                break;
            };
            edges_u.push((te, v_parent, v_curr));
            curr = parent;
        }

        // Trace path 2: bv -> ... -> lca
        let mut path_v: Vec<usize> = Vec::new();
        let mut edges_v: Vec<(usize, usize, usize)> = Vec::new(); // (eid, v_in_curr, v_in_parent)
        let mut curr = bv;
        while curr != lca {
            path_v.push(curr);
            let Some((parent, te, v_curr, v_parent)) = self.tree_step_up(curr) else {
                break;
            };
            edges_v.push((te, v_curr, v_parent));
            curr = parent;
        }

        // Assemble odd cycle starting at `lca`:
        // [lca, path_u.rev()..., path_v...]
        let cycle_len = 1 + path_u.len() + path_v.len();
        let mut children = Vec::with_capacity(cycle_len);
        let mut cycle_edges = Vec::with_capacity(cycle_len);

        children.push(lca);
        for idx in (0..path_u.len()).rev() {
            children.push(path_u[idx]);
            cycle_edges.push(edges_u[idx]);
        }
        // Bridge edge from bu to bv
        cycle_edges.push((eid, u, v));
        for idx in 0..path_v.len() {
            children.push(path_v[idx]);
            cycle_edges.push(edges_v[idx]);
        }

        let base = self.blossoms[lca].base;
        let tree_edge = self.blossoms[lca].tree_edge;

        let mut newly_outer_vertices = Vec::new();
        let mut scratch_leaves = Vec::new();
        for &child in &children {
            let was_inner = self.blossoms[child].label == Label::Inner;
            self.blossoms[child].parent = new_b;
            self.blossom_leaves(child, &mut scratch_leaves);
            for &w in &scratch_leaves {
                self.in_blossom[w] = new_b;
            }
            if was_inner {
                newly_outer_vertices.extend_from_slice(&scratch_leaves);
            }
        }

        self.blossoms[new_b] = BlossomNode {
            parent: UNMATCHED,
            base,
            label: Label::Outer,
            tree_edge,
            dual_z: 0,
            children,
            cycle_edges,
        };

        self.queue.extend(newly_outer_vertices);
    }

    /// Expand a zero-dual `Inner` non-trivial top-level blossom `b`, keeping an
    /// even-length alternating path from its entry sub-blossom to its base sub-blossom.
    fn expand_inner_blossom(&mut self, b: usize) {
        let children = std::mem::take(&mut self.blossoms[b].children);
        let cycle_edges = std::mem::take(&mut self.blossoms[b].cycle_edges);
        let te = self.blossoms[b].tree_edge;
        let len = children.len();

        let mut scratch = Vec::new();
        for &child in &children {
            self.blossoms[child].parent = UNMATCHED;
            self.blossoms[child].label = Label::Unlabeled;
            self.blossoms[child].tree_edge = UNMATCHED;
            self.blossom_leaves(child, &mut scratch);
            for &w in &scratch {
                self.in_blossom[w] = child;
            }
        }

        // Identify which child `c_p` receives `te` from the Outer parent.
        let e = self.edges[te];
        let v_entry = if self.blossoms[self.in_blossom[e.u]].label == Label::Outer {
            e.v
        } else {
            e.u
        };
        let entry_child = self.in_blossom[v_entry];
        let p = children.iter().position(|&c| c == entry_child).unwrap_or(0);

        self.blossoms[children[p]].label = Label::Inner;
        self.blossoms[children[p]].tree_edge = te;

        if p % 2 == 0 {
            // Even index: walk backward p -> p-1 -> p-2 -> ... -> 0
            let steps = p / 2;
            for j in 0..steps {
                let i_out = p - 2 * j - 1;
                let i_next_in = p - 2 * j - 2;
                let out_child = children[i_out];
                let next_in_child = children[i_next_in];
                let out_base = self.blossoms[out_child].base;
                let matched_eid = self.mate_e[out_base];
                self.assign_outer(out_child, matched_eid);
                let unmatched_eid = cycle_edges[i_next_in].0;
                self.blossoms[next_in_child].label = Label::Inner;
                self.blossoms[next_in_child].tree_edge = unmatched_eid;
            }
        } else {
            // Odd index: walk forward p -> p+1 -> ... -> len-1 -> 0
            let steps = (len - p) / 2;
            for j in 0..steps {
                let i_out = p + 2 * j + 1;
                let i_next_in = (p + 2 * j + 2) % len;
                let out_child = children[i_out];
                let next_in_child = children[i_next_in];
                let out_base = self.blossoms[out_child].base;
                let matched_eid = self.mate_e[out_base];
                self.assign_outer(out_child, matched_eid);
                let unmatched_eid = cycle_edges[i_out].0;
                self.blossoms[next_in_child].label = Label::Inner;
                self.blossoms[next_in_child].tree_edge = unmatched_eid;
            }
        }

        self.blossoms[b] = BlossomNode::trivial(0);
        self.free_blossoms.push(b);
    }

    /// Expand an unlabeled zero-dual non-trivial top-level blossom between stages.
    fn expand_unlabeled_blossom(&mut self, b: usize) {
        let children = std::mem::take(&mut self.blossoms[b].children);
        let mut scratch = Vec::new();
        for &child in &children {
            self.blossoms[child].parent = UNMATCHED;
            self.blossoms[child].label = Label::Unlabeled;
            self.blossoms[child].tree_edge = UNMATCHED;
            self.blossom_leaves(child, &mut scratch);
            for &w in &scratch {
                self.in_blossom[w] = child;
            }
        }
        self.blossoms[b] = BlossomNode::trivial(0);
        self.free_blossoms.push(b);
    }

    /// Re-orient the internal matching of blossom `b` so vertex `v` inside `b`
    /// becomes `b.base`.
    fn augment_blossom(&mut self, b: usize, v: usize) {
        if b < self.n {
            return;
        }
        // Climb parent pointers from `v` until reaching an immediate child of `b`.
        let mut t = v;
        while self.blossoms[t].parent != b {
            t = self.blossoms[t].parent;
        }
        if t >= self.n {
            self.augment_blossom(t, v);
        }
        let Some(p) = self.blossoms[b].children.iter().position(|&c| c == t) else {
            return;
        };
        if p == 0 {
            self.blossoms[b].base = v;
            return;
        }

        let len = self.blossoms[b].children.len();
        let pairs = (len - 1) / 2;
        // Collect pairs to re-match along the remaining 2k sub-blossoms of the cycle.
        let mut rematches = Vec::with_capacity(pairs);
        for m in 0..pairs {
            let i1 = (p + 1 + 2 * m) % len;
            let i2 = (p + 2 + 2 * m) % len;
            let c1 = self.blossoms[b].children[i1];
            let c2 = self.blossoms[b].children[i2];
            let (eid, u1, v2) = self.blossoms[b].cycle_edges[i1];
            rematches.push((c1, u1, c2, v2, eid));
        }
        for (c1, u1, c2, v2, eid) in rematches {
            if c1 >= self.n {
                self.augment_blossom(c1, u1);
            }
            if c2 >= self.n {
                self.augment_blossom(c2, v2);
            }
            self.mate_v[u1] = v2;
            self.mate_v[v2] = u1;
            self.mate_e[u1] = eid;
            self.mate_e[v2] = eid;
        }

        self.blossoms[b].children.rotate_left(p);
        self.blossoms[b].cycle_edges.rotate_left(p);
        self.blossoms[b].base = v;
    }

    /// Collect the alternating path from `(start_b, attach_v)` up to its tree root.
    fn collect_tree_augment(
        &self,
        start_b: usize,
        attach_v: usize,
        blossom_rotations: &mut Vec<(usize, usize)>,
        new_matches: &mut Vec<(usize, usize, usize)>,
    ) {
        let mut curr_b = start_b;
        let mut curr_v = attach_v;
        loop {
            blossom_rotations.push((curr_b, curr_v));
            let base = self.blossoms[curr_b].base;
            let mate_vertex = self.mate_v[base];
            if mate_vertex == UNMATCHED {
                // Reached the free root of this alternating tree.
                break;
            }
            let inner_b = self.in_blossom[mate_vertex];
            let Some((next_outer_b, te, w_in, w_out)) = self.tree_step_up(inner_b) else {
                break;
            };
            blossom_rotations.push((inner_b, w_in));
            new_matches.push((w_in, w_out, te));
            curr_b = next_outer_b;
            curr_v = w_out;
        }
    }

    /// Augment the matching along tight edge `eid` connecting `u` and `v` in
    /// two distinct alternating trees.
    fn augment_matching(&mut self, eid: usize, u: usize, v: usize) {
        let bu = self.in_blossom[u];
        let bv = self.in_blossom[v];
        let mut rotations = Vec::new();
        let mut matches = Vec::new();
        self.collect_tree_augment(bu, u, &mut rotations, &mut matches);
        self.collect_tree_augment(bv, v, &mut rotations, &mut matches);
        matches.push((u, v, eid));

        for (b, target_v) in rotations {
            self.augment_blossom(b, target_v);
        }
        for (x, y, me) in matches {
            self.mate_v[x] = y;
            self.mate_v[y] = x;
            self.mate_e[x] = me;
            self.mate_e[y] = me;
        }
    }

    /// Run one primal-dual stage; returns `true` if the matching was augmented.
    fn run_stage(&mut self) -> bool {
        self.queue.clear();
        for b in 0..self.blossoms.len() {
            if self.blossoms[b].parent == UNMATCHED {
                self.blossoms[b].label = Label::Unlabeled;
                self.blossoms[b].tree_edge = UNMATCHED;
            }
        }

        // Every free top-level blossom starts as an Outer root simultaneously.
        let mut has_free = false;
        for v in 0..self.n {
            if self.mate_v[v] == UNMATCHED {
                let b = self.in_blossom[v];
                if self.blossoms[b].label == Label::Unlabeled {
                    self.assign_outer(b, UNMATCHED);
                    has_free = true;
                }
            }
        }
        if !has_free {
            return false;
        }

        let mut head = 0usize;
        loop {
            // 1. Explore tight edges from queued Outer vertices.
            while head < self.queue.len() {
                let u = self.queue[head];
                head += 1;
                let bu = self.in_blossom[u];
                if self.blossoms[bu].label != Label::Outer {
                    continue;
                }

                let deg = self.adj[u].len();
                for idx in 0..deg {
                    let eid = self.adj[u][idx];
                    let v = self.other_endpoint(eid, u);
                    let bv = self.in_blossom[v];
                    if bu == bv {
                        continue;
                    }
                    if self.slack(eid) != 0 {
                        continue;
                    }
                    match self.blossoms[bv].label {
                        Label::Unlabeled => {
                            self.assign_inner(bv, eid);
                        }
                        Label::Inner => {
                            // Already in the alternating forest as Inner; nothing to do.
                        }
                        Label::Outer => {
                            if let Some(lca) = self.find_lca(bu, bv) {
                                self.add_blossom(eid, u, v, lca);
                                // `bu` changed to the new blossom; refresh before continuing.
                                break;
                            } else {
                                self.augment_matching(eid, u, v);
                                return true;
                            }
                        }
                    }
                }
                // If `u` was absorbed into a new blossom during the loop, re-check `u`'s
                // remaining incident edges by re-Compare: actually all edges of `u` still
                // have slack 0 checked if we re-push `u` when `in_blossom[u] != bu`.
                if self.in_blossom[u] != bu {
                    self.queue.push(u);
                }
            }

            // 2. No augmenting path in tight subgraph; compute primal-dual step `delta`.
            // delta1: min dual_y of an Outer vertex (cannot go negative).
            let mut delta_type = 0u8;
            let mut delta = i128::MAX;
            let mut action_edge = UNMATCHED;
            let mut action_blossom = UNMATCHED;

            for v in 0..self.n {
                let b = self.in_blossom[v];
                if self.blossoms[b].label == Label::Outer && self.dual_y[v] < delta {
                    delta = self.dual_y[v];
                    delta_type = 1;
                }
            }

            // delta2 (Outer-Unlabeled: slack) & delta3 (Outer-Outer: slack / 2).
            self.best_edge_to.fill(UNMATCHED);
            for (eid, e) in self.edges.iter().enumerate() {
                let bu = self.in_blossom[e.u];
                let bv = self.in_blossom[e.v];
                if bu == bv {
                    continue;
                }
                let lu = self.blossoms[bu].label;
                let lv = self.blossoms[bv].label;
                if (lu == Label::Outer && lv == Label::Unlabeled)
                    || (lv == Label::Outer && lu == Label::Unlabeled)
                {
                    let s = self.dual_y[e.u] + self.dual_y[e.v] - e.weight;
                    if s < delta || (s == delta && delta_type == 1) {
                        delta = s;
                        delta_type = 2;
                        action_edge = eid;
                    }
                } else if lu == Label::Outer && lv == Label::Outer {
                    let s = (self.dual_y[e.u] + self.dual_y[e.v] - e.weight) / 2;
                    if s < delta || (s == delta && delta_type == 1) {
                        delta = s;
                        delta_type = 3;
                        action_edge = eid;
                    }
                }
            }

            // delta4: min dual_z / 2 over non-trivial Inner top-level blossoms.
            for b in self.n..self.blossoms.len() {
                if self.blossoms[b].parent == UNMATCHED
                    && !self.blossoms[b].children.is_empty()
                    && self.blossoms[b].label == Label::Inner
                {
                    let s = self.blossoms[b].dual_z / 2;
                    if s < delta || (s == delta && delta_type == 1) {
                        delta = s;
                        delta_type = 4;
                        action_blossom = b;
                    }
                }
            }

            // Apply `delta` to all labeled vertices and root non-trivial blossoms.
            if delta > 0 && delta < i128::MAX {
                for v in 0..self.n {
                    match self.blossoms[self.in_blossom[v]].label {
                        Label::Outer => self.dual_y[v] -= delta,
                        Label::Inner => self.dual_y[v] += delta,
                        Label::Unlabeled => {}
                    }
                }
                for b in self.n..self.blossoms.len() {
                    if self.blossoms[b].parent == UNMATCHED && !self.blossoms[b].children.is_empty()
                    {
                        match self.blossoms[b].label {
                            Label::Outer => self.blossoms[b].dual_z += 2 * delta,
                            Label::Inner => self.blossoms[b].dual_z -= 2 * delta,
                            Label::Unlabeled => {}
                        }
                    }
                }
                // Re-scan all Outer vertices in this stage so every newly tight edge is caught.
                head = 0;
            }

            match delta_type {
                1 => {
                    // An Outer vertex reached y_v == 0 with no tight transition left:
                    // dual optimality reached!
                    return false;
                }
                2 => {
                    let e = self.edges[action_edge];
                    let (u, v) = if self.blossoms[self.in_blossom[e.u]].label == Label::Outer {
                        (e.u, e.v)
                    } else {
                        (e.v, e.u)
                    };
                    let bv = self.in_blossom[v];
                    if self.blossoms[bv].label == Label::Unlabeled {
                        self.assign_inner(bv, action_edge);
                    }
                    // Also enqueue `u` so any co-tight edges on `u` are scanned.
                    self.queue.push(u);
                }
                3 => {
                    let e = self.edges[action_edge];
                    let bu = self.in_blossom[e.u];
                    let bv = self.in_blossom[e.v];
                    if bu != bv
                        && self.blossoms[bu].label == Label::Outer
                        && self.blossoms[bv].label == Label::Outer
                    {
                        if let Some(lca) = self.find_lca(bu, bv) {
                            self.add_blossom(action_edge, e.u, e.v, lca);
                        } else {
                            self.augment_matching(action_edge, e.u, e.v);
                            return true;
                        }
                    }
                }
                4 => {
                    if action_blossom != UNMATCHED {
                        self.expand_inner_blossom(action_blossom);
                    }
                }
                _ => return false,
            }
        }
    }

    fn solve(&mut self) -> Vec<usize> {
        for _ in 0..self.n {
            if !self.run_stage() {
                break;
            }
            // Expand any zero-dual unlabeled top-level blossoms to recycle slots.
            loop {
                let mut expanded = false;
                for b in self.n..self.blossoms.len() {
                    if self.blossoms[b].parent == UNMATCHED
                        && !self.blossoms[b].children.is_empty()
                        && self.blossoms[b].dual_z == 0
                    {
                        self.expand_unlabeled_blossom(b);
                        expanded = true;
                    }
                }
                if !expanded {
                    break;
                }
            }
        }
        self.mate_v.clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Exact brute-force solver for small graphs (`n <= 12`).
    fn brute_force_best(
        n: usize,
        edges: &[(usize, usize, i64)],
        max_cardinality: bool,
    ) -> (usize, i64) {
        let mut clean = Vec::new();
        let mut dedup = std::collections::BTreeMap::new();
        for &(a, b, w) in edges {
            if a == b || a >= n || b >= n {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            let entry = dedup.entry(key).or_insert(i64::MIN);
            if w > *entry {
                *entry = w;
            }
        }
        for ((u, v), w) in dedup {
            clean.push((u, v, w));
        }

        fn dfs(
            idx: usize,
            edges: &[(usize, usize, i64)],
            used: u32,
            card: usize,
            weight: i64,
            max_card: bool,
            best: &mut (usize, i64),
        ) {
            let better = if max_card {
                card > best.0 || (card == best.0 && weight > best.1)
            } else {
                weight > best.1
            };
            if better {
                *best = (card, weight);
            }
            if idx == edges.len() {
                return;
            }
            dfs(idx + 1, edges, used, card, weight, max_card, best);
            let (u, v, w) = edges[idx];
            let mask = (1u32 << u) | (1u32 << v);
            if (used & mask) == 0 {
                dfs(
                    idx + 1,
                    edges,
                    used | mask,
                    card + 1,
                    weight + w,
                    max_card,
                    best,
                );
            }
        }

        let mut best = (0usize, 0i64);
        dfs(0, &clean, 0, 0, 0, max_cardinality, &mut best);
        best
    }

    fn evaluate_mate(n: usize, edges: &[(usize, usize, i64)], mate: &[usize]) -> (usize, i64) {
        assert_eq!(mate.len(), n);
        let mut weight_map = std::collections::BTreeMap::new();
        for &(a, b, w) in edges {
            if a == b || a >= n || b >= n {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            let entry = weight_map.entry(key).or_insert(i64::MIN);
            if w > *entry {
                *entry = w;
            }
        }
        let mut pairs = 0usize;
        let mut total = 0i64;
        for u in 0..n {
            let v = mate[u];
            if v == UNMATCHED {
                continue;
            }
            assert!(v < n && v != u, "invalid mate[{u}] = {v}");
            assert_eq!(mate[v], u, "asymmetric mate between {u} and {v}");
            if u < v {
                pairs += 1;
                let w = weight_map
                    .get(&(u, v))
                    .unwrap_or_else(|| panic!("matched nonexistent edge ({u}, {v})"));
                total += *w;
            }
        }
        (pairs, total)
    }

    #[test]
    fn handles_empty_and_single_edge_graphs() {
        assert_eq!(max_weight_matching(0, &[], false), Vec::<usize>::new());
        assert_eq!(max_weight_matching(4, &[], true), vec![UNMATCHED; 4]);
        let mate = max_weight_matching(3, &[(0, 2, 15)], false);
        assert_eq!(mate, vec![2, UNMATCHED, 0]);
    }

    #[test]
    fn triangle_and_pentagon_blossoms_match_optimally() {
        // Triangle + tail where max-weight (1 edge of 100) != max-cardinality (2 edges of 10 + 10 = 20).
        let edges = vec![(0, 1, 10), (1, 2, 100), (2, 0, 10), (0, 3, 10)];
        let mw = max_weight_matching(4, &edges, false);
        assert_eq!(evaluate_mate(4, &edges, &mw), (2, 110)); // (1,2)=100 + (0,3)=10!

        // Where max-weight prefers 1 heavy edge (100) over 2 light edges (10 + 10 = 20):
        let path4 = vec![(0, 1, 10), (1, 2, 100), (2, 3, 10)];
        let mw_path = max_weight_matching(4, &path4, false);
        assert_eq!(evaluate_mate(4, &path4, &mw_path), (1, 100));
        let mc_path = max_weight_matching(4, &path4, true);
        assert_eq!(evaluate_mate(4, &path4, &mc_path), (2, 20));

        // Pentagon + external legs requiring blossom contraction AND subsequent inner-blossom expansion:
        let pentagon = vec![
            (0, 1, 20),
            (1, 2, 20),
            (2, 3, 20),
            (3, 4, 20),
            (4, 0, 20),
            (0, 5, 19),
            (2, 6, 1),
            (3, 7, 18),
        ];
        for max_card in [false, true] {
            let got = max_weight_matching(8, &pentagon, max_card);
            let got_val = evaluate_mate(8, &pentagon, &got);
            let exp_val = brute_force_best(8, &pentagon, max_card);
            assert_eq!(got_val, exp_val, "max_card={max_card}");
        }
    }

    #[test]
    fn exhaustive_agreement_with_brute_force_across_random_graphs() {
        // Deterministic LCG generating dense, sparse, negative/positive, equal-weight,
        // and nested-blossom graphs up to n = 10.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next_u64 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        for trial in 0..250 {
            let n = 2 + (next_u64() as usize % 9); // 2..=10 vertices
            let m = 1 + (next_u64() as usize % (n * (n - 1) / 2 + 1));
            let mut edges = Vec::with_capacity(m);
            for _ in 0..m {
                let u = next_u64() as usize % n;
                let v = next_u64() as usize % n;
                if u == v {
                    continue;
                }
                let w = if trial % 5 == 0 {
                    // Heavy tie-breaking stress test (weights in 1..=3)
                    1 + (next_u64() % 3) as i64
                } else {
                    (next_u64() % 250) as i64 - 30
                };
                edges.push((u, v, w));
            }
            for max_card in [false, true] {
                let mate = max_weight_matching(n, &edges, max_card);
                let got = evaluate_mate(n, &edges, &mate);
                let expected = brute_force_best(n, &edges, max_card);
                if max_card {
                    assert_eq!(
                        got, expected,
                        "mismatch on trial {trial} (n={n}, max_card=true): got {got:?}, expected {expected:?}, edges={edges:?}"
                    );
                } else {
                    assert_eq!(
                        got.1, expected.1,
                        "weight mismatch on trial {trial} (n={n}, max_card=false): got {got:?}, expected {expected:?}, edges={edges:?}"
                    );
                }
            }
        }
    }
}
