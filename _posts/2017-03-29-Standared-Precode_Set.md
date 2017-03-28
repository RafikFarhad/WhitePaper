---
layout: post
title: "Standared-Precode-Set (Corei13)"
date: 2017-03-29 12:56:00 +0600
comments: true
categories:
---

<h3> <a href="#geometry"> Geometry  </a> <br> </h3>
<span style="display:inline-block; width: 40;"> </span> <a href="#fraction"> Fraction Class  </a> <br> 
<span style="display:inline-block; width: 40;"> </span> <a href="#basic">  Basic Geometry  </a> <br> 
<h3>  <a href="#graph"> Graph  </a> <br> </h3>


<hr>
<hr>
<h2> 
<a name="geometry">
  <div id="geometry">
 Geometry 
  </div>
</a>
</h2>

<h5>
<a name="fraction">
  <div id="fraction">
  Fraction 
  </div>
</a>
</h5>

{% highlight c++ %}

/*
    Full featured fraction class
*/

template <class T = long long> struct Fraction {
    T a, b;

    Fraction (T a = T(0), T b = T(1)): a(a), b(b) {
        this->Normalize();
    }

    Fraction (const Fraction& f): a(f.a), b(f.b) {}

    Fraction (double r, int factor = 4) {
        this->b = 1;
        for (int i = 0; i < factor; ++i) {
            this->b *= T(10);
        }
        this->a = T(this->b * r + 0.5);
        this->Normalize();
    }

    Fraction (int a = 0, int b = 1): Fraction(T(a), T(b)) {}

    void Normalize () {
        T d = __gcd (a, b);
        a /= d, b /= d;
        if (b < 0) {
            a = -a, b = -b;
        }
    }

    Fraction operator - () const {
        return Fraction (-this->a, this->b);
    }

    Fraction& operator += (const Fraction& rhs) {
        T a = this->a * rhs.b + this->b * rhs.a, b = this->b * rhs.b;
        this->a = a, this->b = b;
        this->Normalize();
        return *this;
    }

    Fraction& operator += (const T& rhs) {
        this->a += this->b * rhs;
        return *this;
    }

    Fraction& operator -= (const Fraction& rhs) {
        T a = this->a * rhs.b - this->b * rhs.a, b = this->b * rhs.b;
        this->a = a, this->b = b;
        this->Normalize();
        return *this;
    }

    Fraction& operator -= (const T& rhs) {
        this->a -= this->b * rhs;
        return *this;
    }

    Fraction& operator *= (const Fraction& rhs) {
        this->a *= rhs.a, this->b *= rhs.b;
        this->Normalize();
        return *this;
    }

    Fraction& operator *= (const T& rhs) {
        this->a *= rhs;
        this->Normalize();
        return *this;
    }

    Fraction& operator /= (const Fraction& rhs) {
        this->a *= rhs.b, this->b *= rhs.a;
        this->Normalize();
        return *this;
    }

    Fraction& operator /= (const T& rhs) {
        this->b *= rhs;
        this->Normalize();
        return *this;
    }

    friend inline Fraction abs (const Fraction& f) {
        return Fraction (abs(f.a), f.b);
    }

    friend inline Fraction floor (const Fraction& f) {
        return f.a / f.b;
    }

    friend inline Fraction ceil (const Fraction& f) {
        return f.b == 1 ? f.a : f.a / f.b + T(1);
    }

    friend inline Fraction sqrt (const Fraction& f) {
        return Fraction(sqrt(f.a / double(f.b)));
    }

    Fraction& operator %= (const Fraction& rhs) {
        T n = floor(*this / rhs);
        *this -= rhs * n;
        return *this;
    }

    friend inline Fraction operator % (Fraction lhs, const Fraction& rhs) {
        lhs %= rhs;
        return lhs;
    }

    bool operator == (const Fraction& rhs) const {
        return this->a == rhs.a && this->b == rhs.b;
    }

    bool operator == (const T& rhs) const {
        return this->a == rhs && this->b == T(1);
    }

    bool operator < (const Fraction& rhs) const {
        return this->a * rhs.b < this->b * rhs.a;
    }

    bool operator < (const T& rhs) const {
        return this->a < this->b * rhs;
    }

    friend ostream& operator << (ostream& os, Fraction <T>&& f) {
        os << f.a;
        if (f.b != T(1)) {
            os << "/" << f.b;
        }
        return os;
    }

    friend ostream& operator << (ostream& os, Fraction <T>& f) {
        os << f.a;
        if (f.b != T(1)) {
            os << "/" << f.b;
        }
        return os;
    }
};

{% endhighlight %}
<h5>
<a name="basic">
  <div id="basic">
  Basic 
  </div>
</a>
</h5>

{% highlight c++ %}


/*
    Common Geometry Routines
    Shamelessly copied from stanford acm notebook [http://web.stanford.edu/~liszt90/acm/notebook.html]
*/

template <class T = double> struct Point {
    T x, y;

    Point (T x, T y): x(x), y(y) {}

    Point (const Point& p): x(p.x), y(p.y) {}

    Point& operator += (const Point& p) {
        x += p.x, y += p.y;
        return *this;
    }

    friend inline Point operator + (Point lhs, const Point& rhs) {
        lhs += rhs;
        return lhs;
    }

    Point& operator -= (const Point& p) {
        x -= p.x, y -= p.y;
        return *this;
    }

    Point& operator += (const T& c) {
        x += c, y += c;
        return *this;
    }

    Point& operator -= (const T& c) {
        x -= c, y -= c;
        return *this;
    }

    Point& operator *= (const T& c) {
        x *= c, y *= c;
        return *this;
    }

    Point& operator /= (const T& c) {
        x /= c, y /= c;
        return *this;
    }

    bool operator == (const Point& p) const {
        return abs(x - p.x) <= EPS && abs(y - p.y) <= EPS;
    }

    bool operator < (const Point& p) const {
        return x < p.x || (x == p.x && y < p.y);
    }

    friend ostream& operator << (ostream& os, Point <T>&& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }

    friend ostream& operator << (ostream& os, Point <T>& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }

    friend T norm (Point P) {
        return P.x * P.x + P.y * P.y;
    }

    friend Point unit (Point P) {
        return P / norm(P);
    }

    friend Point RotateCCW90 (Point p) {
        return Point(-p.y, p.x);
    }

    friend Point RotateCW90 (Point p) {
        return Point(p.y, -p.x);
    }

    // Rotate clockwise by angle t
    friend Point RotateCCW (Point p, double t) {
        return Point(p.x * cos(t) - p.y * sin(t), p.x * sin(t) + p.y * cos(t));
    }

    friend T dot (Point a, Point b) {
        return a.x * b.x + a.y * b.y;
    }

    friend T cross (Point a, Point b) {
        return a.x * b.y - a.y * b.x;
    }

    friend T dist2 (Point a, Point b) {
        return dot(a - b, a - b);
    }

    // project point c on line (a, b)
    friend Point ProjectPointLine (Point a, Point b, Point c) {
        return a + (b - a) * dot(c - a, b - a) / dot(b - a, b - a);
    }

    // project point c onto line segment [a, b]
    friend Point ProjectPointSegment(Point a, Point b, Point c) {
        T r = dot(b - a, b - a);
        if (abs(r) <= EPS) {
            return a;
        }
        r = dot(c - a, b - a) / r;
        if (r < T(0)) return a;
        if (r > T(1)) return b;
        return a + (b - a) * r;
    }

    // compute distance from c to segment [a, b]
    friend T DistancePointSegment(Point a, Point b, Point c) {
        return sqrt(dist2(c, ProjectOnSegment(a, b, c)));
    }

    // determine if lines (a, b) and (c, d) are parallel or collinear
    friend bool LinesParallel(Point a, Point b, Point c, Point d) {
        return abs(cross(b - a, c - d)) <= EPS;
    }

    friend bool LinesCollinear(Point a, Point b, Point c, Point d) {
        return LinesParallel(a, b, c, d) && abs(cross(a - b, a - c)) <= EPS && abs(cross(c - d, c - a)) <= EPS;
    }

    // determine if line segment [a, b] intersects with line segment [c, d]
    friend bool SegmentsIntersect(Point a, Point b, Point c, Point d) {
        if (LinesCollinear(a, b, c, d)) {
            if (dist2(a, c) <= EPS || dist2(a, d) <= EPS || dist2(b, c) <= EPS || dist2(b, d) <= EPS) {
                return true;
            } else if (dot(c - a, c - b) > T(0) && dot(d - a, d - b) > T(0) && dot(c - b, d - b) > T(0)) {
                return false;
            } else {
                return true;
            }
        } else {
            return cross(d - a, b - a) * cross(c - a, b - a) <= T(0) && cross(a - c, d - c) * cross(b - c, d - c) <= T(0);
        }
    }

    // intersection point between lines (a, b) and (c, d)
    friend Point LineLineIntersection(Point a, Point b, Point c, Point d) {
        b = b - a, d = c - d, c = c - a;
        assert(dot(b, b) >= EPS && dot(d, d) >= EPS);
        return a + b * cross(c, d) / cross(b, d);
    }

    // compute center of circle given three points
    friend Point CircleCenter(Point a, Point b, Point c) {
        b = (a + b) / T(2);
        c = (a + c) / T(2);
        return LineLineIntersection(b, b + RotateCW90(a - b), c, c + RotateCW90(a - c));
    }

    // returns 1(0) if p is a strictly interior (exterior) point of v
    friend bool PointInPolygon(const vector<Point>& v, Point p) {
        bool ret = false;
        for (auto q = v.begin(); q != v.end(); q++) {
            auto r = (next(q) == v.end() ? v.begin() : next(q));
            if ((q->y <= p.y && p.y < r->y || r->y <= p.y && p.y < q->y) &&
                    p.x < q->x + (r->x - q->x) * (p.y - q->y) / (r->y - q->y)) {
                ret = !ret;
            }
        }
        return ret;
    }

    // determine if point is on the boundary of a polygon
    friend bool PointOnPolygon(const vector<Point>& v, Point p) {
        for (auto q = v.begin(); q != v.end(); q++) {
            auto r = (next(q) == v.end() ? v.begin() : next(q));
            if (dist2(ProjectPointSegment(*q, *r, p), p) <= EPS) {
                return true;
            }
        }
        return false;
    }

    // compute intersection of line (a, b) with circle centered at c with radius r > 0
    friend vector <Point> CircleLineIntersection(Point a, Point b, Point c, T r) {
        vector <Point> ret;
        b = b - a;
        a = a - c;
        T A = dot(b, b);
        T B = dot(a, b);
        T C = dot(a, a) - r * r;
        T D = B * B - A * C;
        if (D <= -EPS) {
            return ret;
        }
        ret.push_back(c + a + b * (-B + sqrt(D + EPS)) / A);
        if (D >= EPS)  {
            ret.push_back(c + a + b * (-B - sqrt(D)) / A);
        }
        return ret;
    }

    // compute intersection of circle centered at a with radius r with circle centered at b with radius R
    friend vector<Point> CircleCircleIntersection(Point a, Point b, T r, T R) {
        vector<Point> ret;
        T d = sqrt(dist2(a, b));
        if (d > r + R || d + min(r, R) < max(r, R)) {
            return ret;
        }
        T x = (d * d - R * R + r * r) / (d * T(2));
        T y = sqrt(r * r - x * x);
        Point v = (b - a) / d;
        ret.push_back(a + v * x + RotateCCW90(v) * y);
        if (y > T(0)) {
            ret.push_back(a + v * x - RotateCCW90(v) * y);
        }
        return ret;
    }

    friend T SignedArea(const vector<Point>& v) {
        T area(0);
        for (auto p = v.begin(); p != v.end(); p++) {
            auto q = (next(p) == v.end() ? v.begin() : next(p));
            area = area + (p->x * q->y - q->x * p->y);
        }
        return area / 2.0;
    }

    friend T Area(const vector<Point>& v) {
        return abs(SignedArea(v));
    }

    friend Point Centroid(const vector<Point>& v) {
        Point c(0, 0);
        T scale = 6.0 * SignedArea(v);
        for (auto p = v.begin(); p != v.end(); p++) {
            auto q = (next(p) == v.end() ? v.begin() : next(p));
            c = c + (*p + *q) * (p->x * q->y - q->x * p->y);
        }
        return c / scale;
    }

    // tests whether or not a given polygon (in CW or CCW order) is simple
    friend bool IsSimple(const vector<Point>& v) {
        for (auto p = v.begin(); p != v.end(); p++) {
            for (auto r = next(p); r != v.end(); r++) {
                auto q = (next(p) == v.end() ? v.begin() : next(p));
                auto s = (next(r) == v.end() ? v.begin() : next(r));
                if (p != s && q != r && SegmentsIntersect(*p, *q, *r, *s)) {
                    return false;
                }
            }
        }
        return true;
    }

    // area x 2 of triangle (a, b, c)
    friend T TwiceArea (Point a, Point b, Point c) {
        return cross(a, b) + cross (b, c) + cross (c, a);
    }

    friend T area2 (Point a, Point b, Point c) {
        return cross(a, b) + cross(b, c) + cross(c, a);
    }

    friend void ConvexHull(vector<Point>& v) {
        sort(v.begin(), v.end());
        vector<Point> up, dn;
        for (auto& p : v) {
            while (up.size() > 1 && area2(up[up.size() - 2], up.back(), p) >= 0) {
                up.pop_back();
            }
            while (dn.size() > 1 && area2(dn[dn.size() - 2], dn.back(), p) <= 0) {
                dn.pop_back();
            }
            up.push_back(p);
            dn.push_back(p);
        }
        v = dn;
        v.pop_back();
        reverse(dn.begin(), dn.end());
        for (auto& p : dn) {
            v.push_back(p);
        }
    }
};


{% endhighlight %}


<h2> 
<a name="graph">
  <div id="graph">
   Graph 
  </div>
</a>
</h2>




{% highlight c++ %}

/*
    Implementation of topological sort in directed acyclic graphs

    Running time:
        O(|V| + |E|)

    Usage:
        - add edges by AddEdge()
        - calling Sort() will generate the topological order

    Input:
        - graph, constructed using AddEdge()

    Output:
        - sorted, vector containing the topological order

    Tested Problems:
*/

struct TopologicalSort {
    int n;
    vector <vector <int>> adj;
    vector <int> sorted;

    TopologicalSort (int n): n(n), adj(n) {}

    void AddEdge (int a, int b) {
        adj[a].push_back(b);
    }

    void DFS (int u, vector <bool>& marked) {
        if (marked[u]) {
            return;
        }
        for (auto& v : adj[u]) {
            DFS(v, marked);
        }
        marked[u] = true;
        sorted.push_back(u);
    }

    void Sort () {
        vector <bool> marked(n, false);
        sorted.clear();

        for (int i = 0; i < n; ++i) {
            DFS(i, marked);
        }
    }
};
{% endhighlight %}

{% highlight c++ %}

/*
    Implementation of Tarjan's strongly connected components algorithm.

    Running time:
        O(|V|+|E|)

    Usage:
        - add edges by AddEdge()
        - calling BbuildSCC() will generate the strongly connected components
        - calling Find2SATSolution() will build a solution for the related 2-SAT problem

    Input:
        - graph, constructed using AddEdge()

    Output:
        - components, vector of vectors, where each subvector is a strongly connected component.
        Components are in topological order
        - componentOf, componentOf[i] denotes the component of the node i

    Tested Problems:
*/

struct StronglyConnectedComponents {
    int n;
    int totalComponents;
    vector <vector <int>> adj, components;
    vector <int> idx, componentOf, st, low;
    vector <bool> inStack;

    StronglyConnectedComponents (int n): n(n), adj(n) {}

    void AddEdge (int a, int b) {
        adj[a].push_back(b);
    }

    int DFS (int v, int index) {
        idx[v] = index;
        low[v] = index;
        index += 1;
        st.push_back(v);
        inStack[v] = true;

        for (auto& w : adj[v]) {
            if (idx[w] == -1) {
                index = DFS(w, index);
                low[v] = min(low[v], low[w]);
            } else if (inStack[w]) {
                low[v] = min(low[v], low[w]);
            }
        }

        if (low[v] == idx[v]) {
            int w;
            components.push_back(vector <int>());
            do {
                w = st.back();
                st.pop_back();
                inStack[w] = false;
                componentOf[w] = totalComponents;
                components[totalComponents].push_back(w);
            } while (w != v);
            totalComponents++;
        }
        return index;
    }

    void BuildSCC () {
        totalComponents = 0;
        idx = vector <int>(n, -1), low = vector <int>(n), componentOf = vector <int>(n);
        inStack = vector <bool>(n, false);
        st.clear();

        for (int i = 0; i < n; i++) {
            if (idx[i] == -1) {
                DFS(i, 0);
            }
        }
    }

    /*
        Builds an assignment for a 2-SAT problem

        Input:
            - sol, a vector of integers
            - neg, neg[i] is the negation of i

        Output
            - sol, sol[i] will contain the assignment for i {0: false, 1: true}
    */
    void Find2SATSolution (vector <int>& sol, vector <int>& neg) {
        sol = vector <int>(n, -1);
        for (auto& comp : components) {
            for (auto& j : comp) {
                if (sol[j] == -1) {
                    sol[j] = 1;
                    sol[neg[j]] = 0;
                }
            }
        }
    }
};
{% endhighlight %}

{% highlight c++ %}


/*
    Implementation of Kruskal's minimum spanning tree algorithm

    Running time:
        O(|E|log|V|)

    Usage:
        - add edges by AddEdge()
        - call MST() to generate minimum spanning tree

    Input:
        - n, number of nodes
        - graph, constructed using AddEdge()

    Output:
        - weight of minimum spanning tree
        - mst, if given, mst will contain the edges of the minimum spanning tree

    Tested Problems:
*/

template <class T> struct Kruskal {
    int n;
    vector <pair <T, PI>> edges;
    DisjointSet* D;

    Kruskal (int n): n(n) {}

    void AddEdge (int s, int t, T d) {
        edges.push_back(make_pair(d, make_pair(s, t)));
    }

    T MST (vector <pair <T, PI>>* mst = NULL) {
        T ret = 0;
        D = new DisjointSet(n);
        sort(edges.begin(), edges.end());
        for (auto& e : edges) {
            if (D->Union(e.y.x, e.y.y)) {
                ret += e.x;
                if (mst) {
                    mst->push_back(e);
                }
            }
        }
        return ret;
    }
};
{% endhighlight %}

{% highlight c++ %}


/*
    Implementation of Dijkstra's single source shortest path algorithm

    Running time:
        O(|V|log|V|+|E|)

    Usage:
        - add edges by AddEdge()
        - call BuildTree() to generate shortest path tree

    Input:
        - n, number of nodes
        - directed, true iff the graph is directed
        - graph, constructed using AddEdge()
        - source

    Output:
        - dist, dist[v] is the distance of v from source
        - parent, parent[v] is the parent of v in the tree

    Tested Problems:
*/

template <class T> struct Dijkstra {
    int n;
    bool directed;
    vector <int> parent;
    vector <vector <pair <T, int>>> adj;

    Dijkstra (int n, bool directed = false): n(n), adj(n), directed(directed) {}

    void addEdge (int a, int b, T d) {
        adj[a].push_back(make_pair(d, b));
        if (!directed) {
            adj[b].push_back(make_pair(d, a));
        }
    }

    void BuildTree (int s, vector<T>& dist) {
        dist = vector <T>(n, inf);
        parent = vector <int>(n, -1);
        priority_queue <pair <T, vector<pair <T, int>>>, vector<pair <T, int>>, greater<pair <T, int>>> q;

        dist[s] = 0;
        q.push(make_pair(dist[s], s));
        do {
            auto& u = q.top();
            q.pop();
            for (auto& e : adj[u.y]) {
                if (u.x + e.x < dist[e.y]) {
                    dist[e.y] = u.x + e.x;
                    parent[e.y] = u.y;
                    q.push(make_pair(dist[e.y], e.y));
                }
            }
        } while (!q.empty());
    }
};

{% endhighlight %}

{% highlight c++ %}


/*
    Finds biconnected components, bridges and articulation points in a graph

    Running time:
        O(|V|+|E|)

    Usage:
        - add edges by AddEdge()
        - call BuildBCC() to find all biconnected components, bridges and articulation points

    Input:
        - graph, constructed using AddEdge()

    Output:
        - components, vector of vectors, where each subvector is a biconnected component.
        - bridges, index of the edges which are bridge
        - cutVertices, index of the nodes which are articulation points

    Todo:
        - Implementat 2-connected components finding algorithm

    Tested Problems:
*/

struct BiconnectedComponents {
    int n;
    int totalComponents;
    vector <vector <pair <int, int>>> adj;
    vector <pair <int, int>> edges;
    vector <int> idx, low;
    vector <int> cutVertices, bridges;
    vector <int> st;
    vector <vector <int>> components;

    BiconnectedComponents (int n): n(n), adj(n) {}

    void AddEdge (int a, int b) {
        int i = edges.size();
        adj[a].push_back(make_pair(b, i));
        adj[b].push_back(make_pair(a, i));
        edges.push_back(make_pair(a, b));
    }

    int DFS (PI v, int index) {
        idx[v.x] = index;
        low[v.x] = index;
        index += 1;

        int children = 0;
        bool ap = false;
        for (auto& w : adj[v.x]) {
            if (w.y != v.y) {
                if (idx[w.x] == -1) {
                    st.push_back(w.y);
                    index = DFS(w, index);
                    low[v.x] = min(low[v.x], low[w.x]);
                    if (low[w.x] > idx[v.x]) {
                        bridges.push_back(w.y);
                    }
                    children++;
                    if (low[w.x] >= idx[v.x]) {
                        if (v.y != -1 || children >= 2) {
                            ap = true;
                        }
                        components.push_back(vector <int>());
                        totalComponents++;
                        int u;
                        do {
                            u = st.back();
                            st.pop_back();
                            components.back().push_back(u);
                        } while (u != w.y);
                    }
                } else if (idx[w.x] < idx[v.x]) {
                    st.push_back(w.y);
                    low[v.x] = min(low[v.x], idx[w.x]);
                }
            }
        }
        if (ap) {
            cutVertices.push_back(v.x);
        }
        return index;
    }

    void BuildBCC () {
        idx = vector <int>(n, -1), low = vector <int>(n);
        cutVertices.clear();
        bridges.clear();
        st.clear();
        components.clear();
        totalComponents++;

        for (int i = 0; i < n; i++) {
            if (idx[i] == -1) {
                DFS(make_pair(i, -1), 0);
            }
        }
    }
};

{% endhighlight %}

{% highlight c++ %}


/*
    Implementation of Hopcroft-Karp algorithm of finding maximum matching

    Running time:
        O(|E||V|^{1/2})

    Usage:
        - add edges by AddEdge()
        - indexing is 1-based
        - call Match() to generate the macimum matching
        - MinimumVertexCover() finds a vertex cover of minimum size
        - Maximum independent set is the complement of minimum vertex cover

    Input:
        - graph, constructed using AddEdge()

    Output:
        - number of matching in the maximum matching
        - right, node l of the left is matched with node right[l] of the right, (right[l] = 0 if unmatched)
        - left, node r of the right is matched with node left[r] of the left, (left[r] = 0 if unmatched)

    Todo:
        - Probably convert to 0-base indexing

    Tested Problems:
        - UVA: 12880
*/

struct HopcroftKarp {
    int n, m;
    vector <vector <int>> adj;
    vector <int> right, left;
    vector <int> dist;

    HopcroftKarp (int n, int m): n(n), m(m), adj(n + 1) {}

    void AddEdge (int l, int r) {
        adj[l].push_back(r);
    }

    bool BFS () {
        queue <int> q;
        dist = vector <int>(n + 1, -1);
        for (int l = 1; l <= n; ++l) {
            if (right[l] == 0) {
                dist[l] = 0;
                q.push(l);
            }
        }

        while (!q.empty()) {
            int l = q.front();
            q.pop();
            if (dist[0] == -1 || dist[l] < dist[0]) {
                for (auto& r : adj[l]) if (dist[left[r]] == -1) {
                        dist[left[r]] = dist[l] + 1;
                        q.push(left[r]);
                    }
            }
        }
        return dist[0] != -1;
    }

    bool DFS (int l) {
        if (l != 0) {
            for (auto& r : adj[l]) {
                if (dist[left[r]] == dist[l] + 1 && DFS(left[r])) {
                    left[r] = l;
                    right[l] = r;
                    return true;
                }
            }
            dist[l] = -1;
            return false;
        }
        return true;
    }

    int Match () {
        right = vector <int>(n + 1, 0);
        left = vector <int>(m + 1, 0);
        int ret = 0;
        while (BFS()) {
            for (int l = 1; l <= n; ++l) {
                if (right[l] == 0 && DFS(l)) {
                    ret++;
                }
            }
        }
        return ret;
    }
{% endhighlight %}

{% highlight c++ %}

    /*
        Finds minimum vertex cover

        Running time:
            O(|V|+|E|)

        Output:
            - leftCover, leftCover[l] is true iff node l of the left side is in the minimum vertex cover
            (not in the maximum independent set)
            - leftCover, rightCover[r] is true iff node r of the right side is in the minimum vertex cover
            (not in the maximum independent set)
    */

    void MinimumVertexCover (vector <bool>& leftCover, vector <bool>& rightCover) {
        leftCover = vector <bool>(n + 1, true), rightCover = vector <bool>(m + 1, false);
        queue <int> q;
        dist = vector <int>(n + 1, -1);
        for (int l = 1; l <= n; ++l) {
            if (right[l] == 0) {
                dist[l] = 0;
                q.push(l);
            }
        }

        while (!q.empty()) {
            int l = q.front();
            q.pop();
            leftCover[l] = false;
            if (dist[0] == -1 || dist[l] < dist[0]) {
                for (auto& r : adj[l]) {
                    if (dist[left[r]] == -1) {
                        dist[left[r]] = dist[l] + 1;
                        rightCover[r] = true;
                        q.push(left[r]);
                    }
                }
            }
        }
    }
};
{% endhighlight %}

{% highlight c++ %}


/*
    Implementation of Gale–Shapley stable matching algorithm

    Running time:
        O(n^2)

    Input:
        - n, number of males, number of females
        - maleRank, if male m likes female w more than female w' then maleRank[m][w] < maleRank[m][w']
        - femaleRank, if female w likes male m more than malde m' then femaleRank[w][m] < femaleRank[w][m']

    Output:
        - wife, wife[m] contains wife of male m
        - The matching is male optimal

    Tested Problems:
*/

void StableMatching (const int n, const vector <vector <int>>& maleRank, const vector <vector <int>>& femaleRank, vector <int>& wife) {
    vector <int> freeMen;
    vector <vector <pair <int, int>>> fq(n);
    vector <int> husband(n, -1);

    for (int m = 0; m < n; ++m) {
        for (int w = 0; w < n; ++w) {
            fq[m].push_back(make_pair(maleRank[m][w], w));
        }
        sort(fq[m].begin(), fq[m].end(), greater<PI>());
        freeMen.push_back(m);
    }

    while (!freeMen.empty()) {
        int m = freeMen.back(), w = fq[m].back().y;
        fq[m].pop_back();
        if (husband[w] == -1) {
            husband[w] = m;
            freeMen.pop_back();
        } else if (femaleRank[w][m] < femaleRank[w][husband[w]]) {
            freeMen.pop_back();
            freeMen.push_back(husband[w]);
            husband[w] = m;
        }
    }
    wife = vector <int>(n);
    for (int w = 0; w < n; ++w) {
        wife[husband[w]] = w;
    }
}

{% endhighlight %}

{% highlight c++ %}

/*
    Implementation of euler tour generation algorithm

    Running time:
        O(|V|+|E|)

    Usage:
        - add edges by AddEdge(), use directed = true if the graph is directed
        - StartTour(start) will create a tour from start

    Input:
        - graph, constructed using AddEdge()
        - start, node index to start tour, default is 0

    Output:
        - tour, a list containing the nodes in the tour

    Tested Problems:
*/

struct EulerTour {
    int n, e;
    vector <vector <pair <int, int>>> adj;
    vector <int> ideg, odeg;
    list<int> tour;

    EulerTour(int n): n(n), adj(n), ideg(n), odeg(n), e(0) {}

    void AddEdge(int a, int b, bool directed = false) {
        adj[a].push_back(make_pair(b, e));
        odeg[a]++, ideg[b]++;
        if (!directed) {
            adj[b].push_back(make_pair(a, e));
            odeg[b]++, ideg[a]++;
        }
        e++;
    }

    void StartTour(int start = 0) {
        tour.clear();
        tour.push_back(start);
        vector<vector <pair <int, int>>::iterator> iter;
        for (auto& v : adj) {
            iter.push_back(v.begin());
        }
        vector <bool> visited(e, false);
        for (auto it = tour.begin(); it != tour.end(); it++) {
            int u = *it, v = u;
            auto pos = next(it);
            do {
                while (iter[v] != adj[v].end() && visited[iter[v]->y]) {
                    iter[v]++;
                }
                if (iter[v] == adj[v].end()) {
                    break;
                } else {
                    visited[iter[v]->y] = true;
                    iter[v]++;
                    v = prev(iter[v])->x;
                    tour.insert(pos, v);
                }
            } while (v != u);
        }
    }
};
{% endhighlight %}

{% highlight c++ %}

/*
    Implementation of highest-label push-relabel maximum flow
    with gap relabeling heuristic.

    Running time:
        O(|V|^2|E|^{1/2})

    Usage:
        - add edges by AddEdge()
        - GetMaxFlow(s, t) returns the maximum flow from s to t
        - PreflowToFlow(s, t) converts a preflow network to a flow network
        - GetMinCut returns minimal cuts from source and sink

    Input:
        - graph, constructed using AddEdge()
        - (s, t), (source, sink)

    Output:
        - maximum flow value

    Tested Problems:
*/

template <class T> struct Edge {
    int from, to, index;
    T cap, flow;

    Edge(int from, int to, T cap, T flow, int index): from(from), to(to), cap(cap), flow(flow), index(index) {}
};

template <class T> struct PushRelabel {
    int n;
    vector <vector <Edge <T>>> adj;
    vector <T> excess;
    vector <int> dist, count;
    vector <bool> active;
    vector <vector <int>> B;
    int b;
    queue <int> Q;

    PushRelabel (int n): n(n), adj(n) {}

    void AddEdge (int from, int to, T cap) {
        if (from != to) {
            adj[from].push_back(Edge <T>(from, to, cap, 0, adj[to].size()));
            adj[to].push_back(Edge <T>(to, from, 0, 0, adj[from].size() - 1));
        }
    }

    void Enqueue (int v) {
        if (!active[v] && excess[v] > 0 && dist[v] < n) {
            active[v] = true;
            B[dist[v]].push_back(v);
            b = max(b, dist[v]);
        }
    }

    void Push (Edge <T>& e) {
        T amt = min(excess[e.from], e.cap - e.flow);
        if (dist[e.from] == dist[e.to] + 1 && amt > T(0)) {
            e.flow += amt;
            adj[e.to][e.index].flow -= amt;
            excess[e.to] += amt;
            excess[e.from] -= amt;
            Enqueue(e.to);
        }
    }

    void Gap (int k) {
        for (int v = 0; v < n; v++) if (dist[v] >= k) {
                count[dist[v]]--;
                dist[v] = max(dist[v], n);
                count[dist[v]]++;
                Enqueue(v);
            }
    }

    void Relabel (int v) {
        count[dist[v]]--;
        dist[v] = n;
        for (auto& e : adj[v]) {
            if (e.cap - e.flow > 0) {
                dist[v] = min(dist[v], dist[e.to] + 1);
            }
        }
        count[dist[v]]++;
        Enqueue(v);
    }

    void Discharge(int v) {
        for (auto& e : adj[v]) {
            if (excess[v] > 0) {
                Push(e);
            } else {
                break;
            }
        }

        if (excess[v] > 0) {
            if (count[dist[v]] == 1) {
                Gap(dist[v]);
            } else {
                Relabel(v);
            }
        }
    }

    T GetMaxFlow (int s, int t) {
        dist = vector <int>(n, 0), excess = vector<T>(n, 0), count = vector <int>(n + 1, 0);
        active = vector <bool>(n, false), B = vector <vector <int>>(n);
        b = 0;

        for (int i = 0; i < n; ++i) {
            for (auto& e : adj[i]) {
                e.flow = 0;
            }
        }

        for (auto& e : adj[s]) {
            excess[s] += e.cap;
        }

        count[0] = n;
        Enqueue(s);
        active[t] = true;

        while (b >= 0) {
            if (!B[b].empty()) {
                int v = B[b].back();
                B[b].pop_back();
                active[v] = false;
                Discharge(v);
            } else {
                b--;
            }
        }
        return excess[t];
    }

    void PreflowToFlow (int s, int t) {
        enum color {WHITE, GREY, BLACK};
        vector <color> rank (n, WHITE);
        vector <int> prev (n, -1);
        vector <typename vector <Edge <T>>::iterator> current;
        vector <int> st;
        for (int i = 0; i < n; ++i) {
            current.push_back (adj[i].begin());
        }

        for (int i = 0; i < n; ++i) {
            if (rank[i] == WHITE && excess[i] > 0 && i != s && i != t) {
                int r = i;
                rank[r] = GREY;
                do {
                    while (current[i] != adj[i].end()) {
                        auto e = current[i];
                        if (e->cap == 0 && e->cap - e->flow > 0 && e->to != s && e->to != t) {
                            int j = e->to;
                            if (rank[j] == WHITE) {
                                rank[j] = GREY;
                                prev[j] = i;
                                i = j;
                                break;
                            } else if (rank[j] == GREY) {
                                T amt = e->cap - e->flow;
                                do {
                                    amt = min (amt, current[j]->cap - current[j]->flow);
                                    if (j != i) {
                                        j = current[j]->to;
                                    }
                                } while (j != i);

                                do {
                                    e = current[j];
                                    e->flow += amt;
                                    adj[e->to][e->index].flow -= amt;
                                    j = e->to;
                                } while (j != i);

                                int restart = i;
                                for (j = current[i]->to; j != i; j = e->to) {
                                    e = current[j];
                                    if (rank[j] == WHITE || e->cap - e->flow == 0) {
                                        rank[current[j]->to] = WHITE;
                                        if (rank[j] != WHITE) {
                                            restart = j;
                                        }
                                    }
                                }

                                if (restart != i) {
                                    i = restart;
                                    current[i]++;
                                    break;
                                }
                            }
                        }
                        current[i]++;
                    }
                    if (current[i] == adj[i].end()) {
                        rank[i] = BLACK;
                        if (i != s) {
                            st.push_back(i);
                        }

                        if (i != r) {
                            i = prev[i];
                            current[i]++;
                        } else {
                            break;
                        }
                    }
                } while (true);
            }
        }

        while (!st.empty()) {
            int i = st.back();
            st.pop_back();

            auto e = adj[i].begin();

            while (excess[i] > 0) {
                if (e->cap == 0 && e->cap - e->flow > 0 ) {
                    T amt = min( excess[i], e->cap - e->flow );
                    e->flow += amt;
                    adj[e->to][e->index].flow -= amt;
                    excess[i] -= amt;
                    excess[e->to] += amt;
                }
                e++;
            }
        }
    }
{% endhighlight %}

{% highlight c++ %}

    /*
        Finds minimal cut with source and sink

        Running time:
            O(|V|+|E|)

        Output:
            - cut, cut[u] =
                 0, if u belongs to the minimal cut containing source,
                 1, if u belongs to the minimal cut containing sink,
                -1, otherwise
    */

    T GetMinCut (int s, int t, vector <int>& cut) {
        T ret = GetMaxFlow(s, t);
        PreflowToFlow(s, t);
        cut = vector <int> (n, -1);
        queue<int> Q;

        Q.push(s);
        cut[s] = 0;

        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();

            for (auto& e : adj[u]) {
                if (cut[e.to] == -1 && e.cap - e.flow > 0) {
                    Q.push(e.to);
                    cut[e.to] = 0;
                }
            }
        }

        Q.push(t);
        cut[t] = 1;

        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();

            for (auto& e : adj[u]) {
                if (cut[e.to] == -1 && adj[e.to][e.index].cap - adj[e.to][e.index].flow > 0) {
                    Q.push(e.to);
                    cut[e.to] = 1;
                }
            }
        }

        return ret;
    }
};

{% endhighlight %}
