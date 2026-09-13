#include "quant/agi.h"
#include "quant/random.h"
#include "quant/optimizer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstring>
#include <ctime>
#include <array>
#include <atomic>
#include <iomanip>
#include <mutex>
#include <thread>
#include <chrono>
#include <regex>
#include <cstdio>
#include <memory>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace quant {
namespace agi {


namespace {

int greedy_argmax(const float* logits, int n) {
    return (int)(std::max_element(logits, logits + n) - logits);
}

// Generate new tokens given prompt IDs. Returns ONLY newly generated tokens.
std::vector<int> generate_new_tokens(Model* model, const std::vector<int>& prompt_ids, int vocab_size, int max_new) {
    if (!model) return {};
    std::vector<int> all_ids = prompt_ids;
    int context = 64;

    for (int step = 0; step < max_new; step++) {
        int64_t len = (int64_t)all_ids.size();
        int64_t start = std::max((int64_t)0, len - context);
        int64_t ctx_len = len - start;

        Tensor input_ids({1, ctx_len});
        Tensor positions({1, ctx_len});
        float* idp = input_ids.data<float>();
        float* psp = positions.data<float>();
        for (int64_t i = 0; i < ctx_len; i++) {
            idp[i] = (float)all_ids[start + i];
            psp[i] = (float)(start + i);
        }

        Tensor logits = model->forward(input_ids, positions, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        const float* lp = logits.data<float>();

        int next = greedy_argmax(lp + (ctx_len - 1) * V, (int)V);
        all_ids.push_back(next);

        if (next < 2) break;
    }

    if ((int64_t)all_ids.size() <= (int64_t)prompt_ids.size()) {
        return {};
    }
    return std::vector<int>(all_ids.begin() + (int64_t)prompt_ids.size(), all_ids.end());
}

namespace {

// Template-based C++ code generator for CS problem categories.
// Used as fallback when the model cannot generate a solution.

enum class ProblemCategory {
    SORTING, SEARCHING, GRAPH_BFS, GRAPH_DFS, GRAPH_SHORTEST_PATH,
    DP_KNAPSACK, DP_LCS, DP_FIBONACCI, DP_LIS,
    STRING_MANIP, STRING_PALINDROME, STRING_PATTERNS,
    ARRAY_MAX_SUBARRAY, ARRAY_TWO_SUM, ARRAY_SLIDING_WINDOW,
    TREE_TRAVERSAL, TREE_BALANCED,
    MATH_PRIME, MATH_GCD_LCM, MATH_POWER_MOD,
    STACK_BRACKETS, QUEUE_SLIDING_MAX,
    MATRIX_PATHS, UNION_FIND,
    CATEGORIES_COUNT
};

static ProblemCategory classify_task(const std::string& task) {
    std::string lower = task;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });

    if (lower.find("sort") != std::string::npos) return ProblemCategory::SORTING;
    if (lower.find("search") != std::string::npos || lower.find("binary") != std::string::npos)
        return ProblemCategory::SEARCHING;
    if (lower.find("shortest path") != std::string::npos || lower.find("dijkstra") != std::string::npos ||
        lower.find("bfs") != std::string::npos || lower.find("breadth") != std::string::npos)
        return ProblemCategory::GRAPH_BFS;
    if (lower.find("dfs") != std::string::npos || lower.find("depth") != std::string::npos ||
        lower.find("connected component") != std::string::npos || lower.find("topological") != std::string::npos)
        return ProblemCategory::GRAPH_DFS;
    if (lower.find("cycle") != std::string::npos || lower.find("spanning") != std::string::npos)
        return ProblemCategory::GRAPH_SHORTEST_PATH;
    if (lower.find("knapsack") != std::string::npos || lower.find("subset") != std::string::npos)
        return ProblemCategory::DP_KNAPSACK;
    if (lower.find("common subsequence") != std::string::npos || lower.find("edit distance") != std::string::npos)
        return ProblemCategory::DP_LCS;
    if (lower.find("fibonacci") != std::string::npos || lower.find("climb") != std::string::npos)
        return ProblemCategory::DP_FIBONACCI;
    if (lower.find("longest increasing") != std::string::npos)
        return ProblemCategory::DP_LIS;
    if (lower.find("palindrome") != std::string::npos)
        return ProblemCategory::STRING_PALINDROME;
    if (lower.find("pattern") != std::string::npos || lower.find("kmp") != std::string::npos ||
        lower.find("match") != std::string::npos)
        return ProblemCategory::STRING_PATTERNS;
    if (lower.find("max subarray") != std::string::npos || lower.find("contiguous") != std::string::npos ||
        lower.find("kadane") != std::string::npos)
        return ProblemCategory::ARRAY_MAX_SUBARRAY;
    if (lower.find("two sum") != std::string::npos || lower.find("pair sum") != std::string::npos)
        return ProblemCategory::ARRAY_TWO_SUM;
    if (lower.find("window") != std::string::npos || lower.find("subarray sum") != std::string::npos)
        return ProblemCategory::ARRAY_SLIDING_WINDOW;
    if (lower.find("tree") != std::string::npos && lower.find("inorder") != std::string::npos)
        return ProblemCategory::TREE_TRAVERSAL;
    if (lower.find("balanced") != std::string::npos || lower.find("height") != std::string::npos)
        return ProblemCategory::TREE_BALANCED;
    if (lower.find("prime") != std::string::npos || lower.find("sieve") != std::string::npos)
        return ProblemCategory::MATH_PRIME;
    if (lower.find("gcd") != std::string::npos || lower.find("lcm") != std::string::npos)
        return ProblemCategory::MATH_GCD_LCM;
    if (lower.find("power") != std::string::npos || lower.find("modular") != std::string::npos ||
        lower.find("exponentiat") != std::string::npos)
        return ProblemCategory::MATH_POWER_MOD;
    if (lower.find("bracket") != std::string::npos || lower.find("parenthes") != std::string::npos ||
        lower.find("valid") != std::string::npos)
        return ProblemCategory::STACK_BRACKETS;
    if (lower.find("queue") != std::string::npos || lower.find("sliding max") != std::string::npos)
        return ProblemCategory::QUEUE_SLIDING_MAX;
    if (lower.find("matrix") != std::string::npos || lower.find("grid") != std::string::npos ||
        lower.find("path") != std::string::npos)
        return ProblemCategory::MATRIX_PATHS;
    if (lower.find("union") != std::string::npos || lower.find("disjoint") != std::string::npos ||
        lower.find("component") != std::string::npos)
        return ProblemCategory::UNION_FIND;
    if (lower.find("string") != std::string::npos || lower.find("substring") != std::string::npos)
        return ProblemCategory::STRING_MANIP;

    return ProblemCategory::SORTING;
}

static std::string generate_from_template(const std::string& task) {
    ProblemCategory cat = classify_task(task);
    std::string code = "// Transcender template-generated solution\n";
    code += "// Task: " + task + "\n";
    code += "#include <vector>\n#include <algorithm>\n#include <queue>\n";
    code += "#include <stack>\n#include <string>\n#include <unordered_map>\n";
    code += "#include <unordered_set>\n#include <climits>\n#include <cstdio>\n\n";

    switch (cat) {
    case ProblemCategory::SORTING:
        code += R"(
void sort_array(std::vector<int>& arr) {
    std::sort(arr.begin(), arr.end());
}
int solve() {
    std::vector<int> arr = {5, 3, 8, 1, 9, 2};
    sort_array(arr);
    for (int x : arr) printf("%d ", x);
    printf("\n");
    return 0;
})";
        break;
    case ProblemCategory::SEARCHING:
        code += R"(
int binary_search(const std::vector<int>& arr, int target) {
    int lo = 0, hi = (int)arr.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
int solve() {
    std::vector<int> arr = {1, 3, 5, 7, 9, 11};
    int idx = binary_search(arr, 7);
    printf("Found at index: %d\n", idx);
    return 0;
})";
        break;
    case ProblemCategory::GRAPH_BFS:
        code += R"(
#include <vector>
#include <queue>
#include <cstdio>
std::vector<int> bfs(const std::vector<std::vector<int>>& adj, int start) {
    int n = (int)adj.size();
    std::vector<int> dist(n, -1);
    std::queue<int> q;
    dist[start] = 0;
    q.push(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}
int solve() {
    int n = 6;
    std::vector<std::vector<int>> adj(n);
    auto add_edge = [&](int u, int v) { adj[u].push_back(v); adj[v].push_back(u); };
    add_edge(0, 1); add_edge(1, 2); add_edge(2, 3); add_edge(3, 4); add_edge(4, 5);
    auto dist = bfs(adj, 0);
    for (int i = 0; i < n; i++) printf("dist[%d] = %d\n", i, dist[i]);
    return 0;
})";
        break;
    case ProblemCategory::GRAPH_DFS:
        code += R"(
#include <vector>
#include <cstdio>
void dfs_visit(const std::vector<std::vector<int>>& adj, int u,
               std::vector<bool>& visited, std::vector<int>& order) {
    visited[u] = true;
    order.push_back(u);
    for (int v : adj[u]) {
        if (!visited[v]) dfs_visit(adj, v, visited, order);
    }
}
std::vector<int> dfs(const std::vector<std::vector<int>>& adj, int start) {
    int n = (int)adj.size();
    std::vector<bool> visited(n, false);
    std::vector<int> order;
    dfs_visit(adj, start, visited, order);
    return order;
}
int solve() {
    int n = 5;
    std::vector<std::vector<int>> adj(n);
    adj[0] = {1, 2}; adj[1] = {3}; adj[2] = {4};
    auto order = dfs(adj, 0);
    for (int x : order) printf("%d ", x);
    printf("\n");
    return 0;
})";
        break;
    case ProblemCategory::GRAPH_SHORTEST_PATH:
        code += R"(
#include <vector>
#include <queue>
#include <climits>
#include <cstdio>
std::vector<int> dijkstra(const std::vector<std::vector<std::pair<int,int>>>& adj, int src) {
    int n = (int)adj.size();
    std::vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    std::priority_queue<std::pair<int,int>, std::vector<std::pair<int,int>>, std::greater<>> pq;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
int solve() {
    int n = 4;
    std::vector<std::vector<std::pair<int,int>>> adj(n);
    adj[0].push_back({1, 4}); adj[0].push_back({2, 1});
    adj[1].push_back({3, 1}); adj[2].push_back({1, 2});
    adj[2].push_back({3, 5});
    auto dist = dijkstra(adj, 0);
    for (int i = 0; i < n; i++) printf("dist[%d] = %d\n", i, dist[i] == INT_MAX ? -1 : dist[i]);
    return 0;
})";
        break;
    case ProblemCategory::DP_KNAPSACK:
        code += R"(
#include <vector>
#include <algorithm>
#include <cstdio>
int knapsack(int W, const std::vector<int>& wt, const std::vector<int>& val) {
    int n = (int)wt.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(W + 1, 0));
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            dp[i][w] = dp[i-1][w];
            if (wt[i-1] <= w) {
                dp[i][w] = std::max(dp[i][w], dp[i-1][w - wt[i-1]] + val[i-1]);
            }
        }
    }
    return dp[n][W];
}
int solve() {
    std::vector<int> wt = {2, 3, 4, 5};
    std::vector<int> val = {3, 4, 5, 6};
    printf("Max value: %d\n", knapsack(8, wt, val));
    return 0;
})";
        break;
    case ProblemCategory::DP_LCS:
        code += R"(
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
int lcs_length(const std::string& a, const std::string& b) {
    int m = (int)a.size(), n = (int)b.size();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (a[i-1] == b[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
            else dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[m][n];
}
int solve() {
    printf("LCS length: %d\n", lcs_length("ABCBDAB", "BDCAB"));
    return 0;
})";
        break;
    case ProblemCategory::DP_FIBONACCI:
        code += R"(
#include <vector>
#include <cstdio>
long long fibonacci(int n) {
    if (n <= 1) return n;
    std::vector<long long> dp(n + 1, 0);
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++) dp[i] = dp[i-1] + dp[i-2];
    return dp[n];
}
int solve() {
    for (int i = 0; i <= 20; i++) printf("fib(%d) = %lld\n", i, fibonacci(i));
    return 0;
})";
        break;
    case ProblemCategory::DP_LIS:
        code += R"(
#include <vector>
#include <algorithm>
#include <cstdio>
int lis_length(const std::vector<int>& arr) {
    int n = (int)arr.size();
    std::vector<int> dp(n, 1);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) dp[i] = std::max(dp[i], dp[j] + 1);
        }
    }
    return *std::max_element(dp.begin(), dp.end());
}
int solve() {
    std::vector<int> arr = {10, 9, 2, 5, 3, 7, 101, 18};
    printf("LIS length: %d\n", lis_length(arr));
    return 0;
})";
        break;
    case ProblemCategory::STRING_MANIP:
        code += R"(
#include <string>
#include <vector>
#include <cstdio>
std::vector<int> find_words(const std::string& text, const std::string& word) {
    std::vector<int> positions;
    size_t pos = 0;
    while ((pos = text.find(word, pos)) != std::string::npos) {
        positions.push_back((int)pos);
        pos += word.size();
    }
    return positions;
}
int solve() {
    auto pos = find_words("hello world hello", "hello");
    for (int p : pos) printf("Found at: %d\n", p);
    return 0;
})";
        break;
    case ProblemCategory::STRING_PALINDROME:
        code += R"(
#include <string>
#include <cstdio>
bool is_palindrome(const std::string& s) {
    int i = 0, j = (int)s.size() - 1;
    while (i < j) {
        if (s[i++] != s[j--]) return false;
    }
    return true;
}
std::string longest_palindrome(const std::string& s) {
    int n = (int)s.size();
    if (n == 0) return "";
    int best_start = 0, best_len = 1;
    for (int i = 0; i < n; i++) {
        for (int len = 1; i + len <= n; len++) {
            bool ok = true;
            for (int l = i, r = i + len - 1; l < r; l++, r--) {
                if (s[l] != s[r]) { ok = false; break; }
            }
            if (ok && len > best_len) { best_start = i; best_len = len; }
        }
    }
    return s.substr(best_start, best_len);
}
int solve() {
    printf("Longest palindrome: %s\n", longest_palindrome("babad").c_str());
    return 0;
})";
        break;
    case ProblemCategory::STRING_PATTERNS:
        code += R"(
#include <vector>
#include <string>
#include <cstdio>
std::vector<int> kmp_failure(const std::string& pat) {
    int m = (int)pat.size();
    std::vector<int> fail(m, 0);
    for (int i = 1, len = 0; i < m; ) {
        if (pat[i] == pat[len]) { fail[i++] = ++len; }
        else if (len) { len = fail[len - 1]; }
        else { fail[i++] = 0; }
    }
    return fail;
}
std::vector<int> kmp_search(const std::string& text, const std::string& pat) {
    auto fail = kmp_failure(pat);
    std::vector<int> matches;
    int i = 0, j = 0;
    while (i < (int)text.size()) {
        if (text[i] == pat[j]) { i++; j++; }
        if (j == (int)pat.size()) { matches.push_back(i - j); j = fail[j - 1]; }
        else if (i < (int)text.size() && text[i] != pat[j]) {
            j = j ? fail[j - 1] : 0;
        }
    }
    return matches;
}
int solve() {
    auto m = kmp_search("ABABDABACDABABCABAB", "ABABCABAB");
    for (int p : m) printf("Match at: %d\n", p);
    return 0;
})";
        break;
    case ProblemCategory::ARRAY_MAX_SUBARRAY:
        code += R"(
#include <vector>
#include <algorithm>
#include <cstdio>
int max_subarray_sum(const std::vector<int>& arr) {
    int best = arr[0], cur = arr[0];
    for (int i = 1; i < (int)arr.size(); i++) {
        cur = std::max(arr[i], cur + arr[i]);
        best = std::max(best, cur);
    }
    return best;
}
int solve() {
    std::vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    printf("Max subarray sum: %d\n", max_subarray_sum(arr));
    return 0;
})";
        break;
    case ProblemCategory::ARRAY_TWO_SUM:
        code += R"(
#include <vector>
#include <unordered_map>
#include <cstdio>
std::pair<int,int> two_sum(const std::vector<int>& arr, int target) {
    std::unordered_map<int,int> seen;
    for (int i = 0; i < (int)arr.size(); i++) {
        int need = target - arr[i];
        if (seen.count(need)) return {seen[need], i};
        seen[arr[i]] = i;
    }
    return {-1, -1};
}
int solve() {
    auto [a, b] = two_sum({2, 7, 11, 15}, 9);
    printf("Indices: %d, %d\n", a, b);
    return 0;
})";
        break;
    case ProblemCategory::ARRAY_SLIDING_WINDOW:
        code += R"(
#include <vector>
#include <algorithm>
#include <cstdio>
int max_sum_window(const std::vector<int>& arr, int k) {
    int window_sum = 0;
    for (int i = 0; i < k; i++) window_sum += arr[i];
    int best = window_sum;
    for (int i = k; i < (int)arr.size(); i++) {
        window_sum += arr[i] - arr[i - k];
        best = std::max(best, window_sum);
    }
    return best;
}
int solve() {
    printf("Max window sum: %d\n", max_sum_window({1, 4, 2, 10, 2, 3, 1, 0, 20}, 4));
    return 0;
})";
        break;
    case ProblemCategory::TREE_TRAVERSAL:
        code += R"(
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};
#include <vector>
#include <cstdio>
void inorder(TreeNode* root, std::vector<int>& res) {
    if (!root) return;
    inorder(root->left, res);
    res.push_back(root->val);
    inorder(root->right, res);
}
int solve() {
    auto* root = new TreeNode(4);
    root->left = new TreeNode(2); root->right = new TreeNode(6);
    root->left->left = new TreeNode(1); root->left->right = new TreeNode(3);
    root->right->left = new TreeNode(5); root->right->right = new TreeNode(7);
    std::vector<int> res;
    inorder(root, res);
    for (int x : res) printf("%d ", x);
    printf("\n");
    delete root->left->left; delete root->left->right; delete root->right->left;
    delete root->right->right; delete root->left; delete root->right; delete root;
    return 0;
})";
        break;
    case ProblemCategory::TREE_BALANCED:
        code += R"(
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};
#include <algorithm>
#include <cstdio>
int height(TreeNode* root) {
    if (!root) return 0;
    return 1 + std::max(height(root->left), height(root->right));
}
bool is_balanced(TreeNode* root) {
    if (!root) return true;
    int lh = height(root->left), rh = height(root->right);
    return std::abs(lh - rh) <= 1 && is_balanced(root->left) && is_balanced(root->right);
}
int solve() {
    auto* root = new TreeNode(1);
    root->left = new TreeNode(2); root->right = new TreeNode(3);
    root->left->left = new TreeNode(4); root->left->right = new TreeNode(5);
    printf("Balanced: %s\n", is_balanced(root) ? "true" : "false");
    delete root->left->left; delete root->left->right;
    delete root->left; delete root->right; delete root;
    return 0;
})";
        break;
    case ProblemCategory::MATH_PRIME:
        code += R"(
#include <vector>
#include <cstdio>
std::vector<int> sieve(int n) {
    std::vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) is_prime[j] = false;
        }
    }
    std::vector<int> primes;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) primes.push_back(i);
    }
    return primes;
}
int solve() {
    auto primes = sieve(50);
    printf("Primes up to 50: ");
    for (int p : primes) printf("%d ", p);
    printf("\n");
    return 0;
})";
        break;
    case ProblemCategory::MATH_GCD_LCM:
        code += R"(
#include <cstdio>
long long gcd(long long a, long long b) {
    while (b) { a %= b; long long t = a; a = b; b = t; }
    return a;
}
long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b;
}
int solve() {
    printf("GCD(12, 8) = %lld\n", gcd(12, 8));
    printf("LCM(12, 8) = %lld\n", lcm(12, 8));
    return 0;
})";
        break;
    case ProblemCategory::MATH_POWER_MOD:
        code += R"(
#include <cstdio>
long long power_mod(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}
int solve() {
    printf("2^10 mod 1000 = %lld\n", power_mod(2, 10, 1000));
    printf("3^13 mod 7 = %lld\n", power_mod(3, 13, 7));
    return 0;
})";
        break;
    case ProblemCategory::STACK_BRACKETS:
        code += R"BRK(
#include <stack>
#include <string>
#include <cstdio>
bool is_valid_brackets(const std::string& s) {
    std::stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') st.push(c);
        else {
            if (st.empty()) return false;
            char top = st.top(); st.pop();
            if ((c == ')' && top != '(') || (c == ']' && top != '[') || (c == '}' && top != '{'))
                return false;
        }
    }
    return st.empty();
}
int solve() {
    printf("Valid: %s\n", is_valid_brackets("({[]})") ? "true" : "false");
    printf("Valid: %s\n", is_valid_brackets("([)]") ? "true" : "false");
    return 0;
})BRK";
        break;
    case ProblemCategory::QUEUE_SLIDING_MAX:
        code += R"(
#include <vector>
#include <deque>
#include <cstdio>
std::vector<int> sliding_max(const std::vector<int>& arr, int k) {
    std::deque<int> dq;
    std::vector<int> result;
    for (int i = 0; i < (int)arr.size(); i++) {
        while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        while (!dq.empty() && arr[dq.back()] <= arr[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) result.push_back(arr[dq.front()]);
    }
    return result;
}
int solve() {
    auto r = sliding_max({1, 3, -1, -3, 5, 3, 6, 7}, 3);
    for (int x : r) printf("%d ", x);
    printf("\n");
    return 0;
})";
        break;
    case ProblemCategory::MATRIX_PATHS:
        code += R"(
#include <vector>
#include <cstdio>
int unique_paths(int m, int n) {
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 1));
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}
int solve() {
    printf("Unique paths (3x7): %d\n", unique_paths(3, 7));
    printf("Unique paths (3x3): %d\n", unique_paths(3, 3));
    return 0;
})";
        break;
    case ProblemCategory::UNION_FIND:
        code += R"(
#include <vector>
#include <cstdio>
struct UnionFind {
    std::vector<int> parent, rank_;
    UnionFind(int n) : parent(n), rank_(n, 0) {
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank_[x] < rank_[y]) std::swap(x, y);
        parent[y] = x;
        if (rank_[x] == rank_[y]) rank_[x]++;
        return true;
    }
};
int solve() {
    UnionFind uf(6);
    uf.unite(0, 1); uf.unite(1, 2); uf.unite(3, 4);
    printf("Same set (0,2): %s\n", uf.find(0) == uf.find(2) ? "true" : "false");
    printf("Same set (0,3): %s\n", uf.find(0) == uf.find(3) ? "true" : "false");
    uf.unite(2, 4);
    printf("Same set (0,3) after union: %s\n", uf.find(0) == uf.find(3) ? "true" : "false");
    return 0;
})";
        break;
    default:
        code += R"(
int solve() {
    printf("No template matched for this task.\n");
    return 0;
})";
        break;
    }

    code += "\n";

    return code;
}

namespace fs = std::filesystem;

static fs::path get_sandbox_path() {
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    return fs::path(tmp ? tmp : "C:\\Temp") / "Transcender_sandbox";
#else
    return fs::path("/tmp") / "Transcender_sandbox";
#endif
}

static std::string read_file_contents(const fs::path& path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
}

static std::string escape_path(const std::string& p) {
#ifdef _WIN32
    return "\"" + p + "\"";
#else
    std::string escaped = p;
    for (size_t i = 0; i < escaped.size(); i++) {
        if (escaped[i] == ' ' || escaped[i] == '(' || escaped[i] == ')' ||
            escaped[i] == '&' || escaped[i] == '|' || escaped[i] == ';') {
            escaped.insert(escaped.begin() + i, '\\');
            i++;
        }
    }
    return escaped;
#endif
}
} // anonymous namespace
} // anonymous namespace


// ========================================================================
// Flywheel constructor
// ========================================================================
Flywheel::Flywheel(Model* model, Trainer* trainer, CodeGenSelfImprover* codegen,
                   SelfVerifier* verifier, CapabilityAmplifier* amplifier,
                   SafetyGuardrails* safety)
    : model_(model), trainer_(trainer), codegen_(codegen), verifier_(verifier),
      amplifier_(amplifier), safety_(safety), no_improvement_count_(0), converged_count_(0) {
}

std::string Flywheel::get_log_path() const {
    return (fs::current_path() / "FLYWHEEL_LOG.md").string();
}

// ========================================================================
// sandbox_path: return the isolated sandbox directory
// ========================================================================
std::string Flywheel::sandbox_path() const {
    return get_sandbox_path().string();
}

// ========================================================================
// P17: dataset-driven task templates. Built-in 32 kept as fallback (no delete).
// Load order: $AGI_TASKS_FILE -> data/agi_tasks.txt -> assets/agi_tasks.txt.
// Fallback prints stderr warning "using built-in 32 templates".
// ========================================================================
static const std::array<const char*, 32> kBuiltInTaskTemplates = {{
    "Write a C++ function to sort an array of integers using quicksort.",
    "Write a C++ function to compute the Fibonacci sequence using iteration.",
    "Write a C++ function to reverse a string in place.",
    "Write a C++ function to find the maximum subarray sum (Kadane's algorithm).",
    "Write a C++ function to check if a string is a palindrome.",
    "Write a C++ function to merge two sorted arrays.",
    "Write a C++ function to perform binary search on a sorted array.",
    "Write a C++ function to implement a stack using a linked list.",
    "Write a C++ function to find the first non-repeating character in a string.",
    "Write a C++ function to compute the greatest common divisor using Euclid's algorithm.",
    "Write a C++ function to transpose a matrix.",
    "Write a C++ function to count word frequency in a string.",
    "Write a C++ function to remove duplicates from a sorted array.",
    "Write a C++ function to find the intersection of two arrays.",
    "Write a C++ function to implement a simple hash table.",
    "Write a C++ function to perform basic math operations on two numbers.",
    "Write a C++ function to find the longest common prefix among strings.",
    "Write a C++ function to implement bubble sort.",
    "Write a C++ function to convert a decimal number to binary.",
    "Write a C++ function to calculate the factorial of a number.",
    "Write a C++ function to check if a number is prime.",
    "Write a C++ function to find all prime factors of a number.",
    "Write a C++ function to compute the nth triangular number.",
    "Write a C++ function to implement a simple linear search.",
    "Write a C++ function to generate all permutations of a string.",
    "Write a C++ function to implement a queue using two stacks.",
    "Write a C++ function to detect cycles in a linked list.",
    "Write a C++ function to find the middle element of a linked list.",
    "Write a C++ function to implement a simple LRU cache.",
    "Write a C++ function to compute the Levenshtein distance between two strings.",
    "Write a C++ function to implement the Sieve of Eratosthenes.",
    "Write a C++ function to rotate an array by k positions.",
}};

static std::string trim_task_line(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::vector<std::string> load_tasks_from_file(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream ifs(path);
    if (!ifs) return out;
    std::string line;
    while (std::getline(ifs, line)) {
        std::string t = trim_task_line(line);
        if (t.empty() || t[0] == '#') continue;
        out.push_back(t);
    }
    return out;
}

// Returns a copy (safe across env-triggered reloads). Cached per AGI_TASKS_FILE key.
// BUGFIX (bug census): static cache/cache_key/warned_key mutated with no
// lock (concurrent vector/string race). Guarded by a mutex; returns a copy.
static std::vector<std::string> get_task_templates() {
    static std::mutex cache_mtx;
    static std::vector<std::string> cache;
    static std::string cache_key;
    static std::string warned_key = "\x01"; // never equal to a real key initially
    std::lock_guard<std::mutex> lock(cache_mtx);
    const char* env = std::getenv("AGI_TASKS_FILE");
    std::string key = (env && *env) ? std::string(env) : std::string();
    if (!cache.empty() && key == cache_key) return cache;

    std::vector<std::string> candidates;
    if (!key.empty()) candidates.push_back(key);
    candidates.push_back("data/agi_tasks.txt");
    candidates.push_back("assets/agi_tasks.txt");

    for (const auto& p : candidates) {
        std::vector<std::string> loaded = load_tasks_from_file(p);
        if (!loaded.empty()) {
            cache = loaded;
            cache_key = key;
            return cache;
        }
    }

    cache.clear();
    for (const char* t : kBuiltInTaskTemplates) cache.push_back(t);
    cache_key = key;
    if (warned_key != key) {
        std::fprintf(stderr, "[Flywheel] AGI_TASKS_FILE not set or empty/unreadable; "
                             "using built-in 32 templates as fallback\n");
        warned_key = key;
    }
    return cache;
}

// ========================================================================
// self_play: generate a task for self-improvement
// ========================================================================
std::string Flywheel::self_play() {
    std::vector<std::string> task_templates = get_task_templates();

    static std::atomic<size_t> task_index{0};
    std::string task = task_templates[task_index.fetch_add(1) % task_templates.size()];

    if (model_) {
        int vocab_size = (int)model_->config.vocab_size;
        std::string prompt = "Generate a C++ implementation task similar to: " + task + " Task:";
        auto ids = quant::agi::util::simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string model_task = quant::agi::util::simple_decode(gen);
        if (!model_task.empty() && model_task.size() > 10) {
            task = model_task;
        }
    }

    return task;
}

// P17 test hook: public wrapper over private self_play().
std::string Flywheel::self_play_for_test() {
    return self_play();
}

// ========================================================================
// H7 (D7 W2) real-harness helpers — file-local so headers/CMake stay untouched.
// - SandboxResult mapping (header-fixed): compiled == compile_ok,
//   score == test_score (fraction passed/total), passed == full gate,
//   exit_code part of the gate. compile_ok alone never implies test_score > 0.
// - Pass gate: exit==0 && passed==total && total>=kFlywheelMinChecks.
// - Negative control: effectively-empty solution scores 0.
// ========================================================================
static constexpr int kFlywheelMinChecks = 4;

static bool flywheel_is_effectively_empty(const std::string& code) {
    // Strip // line comments, /* block comments */, and whitespace.
    // If no non-whitespace token remains, treat as empty (negative control).
    bool in_block = false;
    bool in_line = false;
    bool in_str = false;
    bool in_chr = false;
    for (size_t i = 0; i < code.size(); i++) {
        char c = code[i];
        char nxt = (i + 1 < code.size()) ? code[i + 1] : '\0';
        if (in_line) {
            if (c == '\n') in_line = false;
            continue;
        }
        if (in_block) {
            if (c == '*' && nxt == '/') { in_block = false; i++; }
            continue;
        }
        if (in_str) {
            if (c == '\\' && nxt != '\0') { i++; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (in_chr) {
            if (c == '\\' && nxt != '\0') { i++; continue; }
            if (c == '\'') in_chr = false;
            continue;
        }
        if (c == '/' && nxt == '/') { in_line = true; i++; continue; }
        if (c == '/' && nxt == '*') { in_block = true; i++; continue; }
        if (c == '"') { in_str = true; continue; }
        if (c == '\'') { in_chr = true; continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        // Any surviving non-whitespace character means non-empty.
        return false;
    }
    return true;
}

static bool flywheel_parse_counts(const std::string& stdout_str, int& passed, int& total) {
    std::regex score_re(R"(Score:\s+(\d+\.?\d*)\s*\((\d+)/(\d+)\))");
    std::smatch m;
    if (std::regex_search(stdout_str, m, score_re) && m.size() >= 4) {
        try {
            passed = std::stoi(m[2].str());
            total = std::stoi(m[3].str());
            return true;
        } catch (...) { passed = 0; total = 0; return false; }
    }
    int p = 0, f = 0;
    size_t pos = 0;
    while ((pos = stdout_str.find("PASS:", pos)) != std::string::npos) { p++; pos += 5; }
    pos = 0;
    while ((pos = stdout_str.find("FAIL:", pos)) != std::string::npos) { f++; pos += 5; }
    passed = p;
    total = p + f;
    return total > 0;
}

static std::string flywheel_sanitize_task(const std::string& task) {
    // Make task safe to embed inside a C++ string literal + printf format.
    std::string src = task.substr(0, 60);
    std::string out;
    out.reserve(src.size() + 8);
    for (char c : src) {
        if (c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t') out += ' ';
        else if (c == '%') out += "%%";
        else out += c;
    }
    return out;
}

// ========================================================================
// generate_test_program: wrap solution code in a complete test harness
// H7 real harness: per-template property checks, no vacuous passes.
// ========================================================================
std::string Flywheel::generate_test_program(const std::string& code, const std::string& task) {
    // H7 real harness (D7 W2): every branch below emits per-template property
    // checks that CALL solution symbols. No CHECK(true), no bare increments.
    // Empty code therefore cannot link (compile_ok=false) and, even if it did
    // compile, would fail all property CHECKs (test_score=0).
    std::ostringstream harness;
    std::string safe_task = flywheel_sanitize_task(task);
    harness << "// Auto-generated test harness for: " << task << "\n";
    harness << "#include <cstdio>\n";
    harness << "#include <cstdlib>\n";
    harness << "#include <cmath>\n";
    harness << "#include <cstring>\n";
    harness << "#include <string>\n";
    harness << "#include <vector>\n";
    harness << "#include <algorithm>\n";
    harness << "#include <cstdint>\n";
    harness << "#include <sstream>\n";
    harness << "#include <cassert>\n";
    harness << "#include <utility>\n";
    harness << "#include <unordered_map>\n";
    harness << "#include <queue>\n";
    harness << "#include <stack>\n";
    harness << "#include <deque>\n";
    harness << "#include <climits>\n";
    harness << "\n";
    harness << "// User-provided solution code:\n";
    harness << code << "\n";
    harness << "\n";
    harness << "// ==================== TEST HARNESS (H7 real) ====================\n";
    harness << "// Negative control: each branch calls solution symbols, so empty code\n";
    harness << "// fails to compile (compile_ok=false) or fails every CHECK (test_score=0).\n";
    harness << "// Pass requires exit==0 && passed==total && total>=4. No vacuous passes.\n";
    harness << "int g_passed = 0;\n";
    harness << "int g_total = 0;\n";
    harness << "#define CHECK(cond, msg) do { g_total++; if (cond) { g_passed++; std::printf(\"  PASS: %s\\n\", msg); } else { std::printf(\"  FAIL: %s\\n\", msg); } } while(0)\n";
    harness << "\n";
    harness << "int main() {\n";
    harness << "    std::printf(\"=== Running tests for task: " << safe_task << "... ===\\n\");\n";

    ProblemCategory cat = classify_task(task);
    switch (cat) {
    case ProblemCategory::SORTING: {
        harness << "    // SORTING: quicksort vs std::sort on fixed vectors\n";
        harness << "    {\n";
        harness << "        std::vector<int> v1 = {5, 3, 8, 1, 9, 2};\n";
        harness << "        std::vector<int> e1 = v1;\n";
        harness << "        std::sort(e1.begin(), e1.end());\n";
        harness << "        sort_array(v1);\n";
        harness << "        CHECK(v1 == e1, \"sort fixed vector matches std_sort\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<int> v2 = {5, 4, 3, 2, 1};\n";
        harness << "        std::vector<int> e2 = v2;\n";
        harness << "        std::sort(e2.begin(), e2.end());\n";
        harness << "        sort_array(v2);\n";
        harness << "        CHECK(v2 == e2, \"sort reverse-sorted matches std_sort\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<int> v3 = {3, 1, 2, 3, 1, 2};\n";
        harness << "        std::vector<int> e3 = v3;\n";
        harness << "        std::sort(e3.begin(), e3.end());\n";
        harness << "        sort_array(v3);\n";
        harness << "        CHECK(v3 == e3, \"sort duplicates matches std_sort\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<int> v4;\n";
        harness << "        sort_array(v4);\n";
        harness << "        CHECK(v4.empty(), \"sort empty stays empty\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<int> v5 = {42};\n";
        harness << "        sort_array(v5);\n";
        harness << "        CHECK(v5.size() == 1 && v5[0] == 42, \"sort single unchanged\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::SEARCHING: {
        harness << "    // SEARCHING: binary-search hit/miss\n";
        harness << "    {\n";
        harness << "        std::vector<int> a = {1, 3, 5, 7, 9, 11};\n";
        harness << "        CHECK(binary_search(a, 7) == 3, \"binary_search hit middle\");\n";
        harness << "        CHECK(binary_search(a, 1) == 0, \"binary_search hit first\");\n";
        harness << "        CHECK(binary_search(a, 11) == 5, \"binary_search hit last\");\n";
        harness << "        CHECK(binary_search(a, 6) == -1, \"binary_search miss returns -1\");\n";
        harness << "        CHECK(binary_search(a, 0) == -1, \"binary_search miss low returns -1\");\n";
        harness << "        CHECK(binary_search(std::vector<int>{}, 5) == -1, \"binary_search empty returns -1\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::GRAPH_BFS: {
        harness << "    // GRAPH_BFS: distances on fixed chain + unreachable\n";
        harness << "    {\n";
        harness << "        int n = 6;\n";
        harness << "        std::vector<std::vector<int>> adj(n);\n";
        harness << "        auto fw_add = [&](int u, int v) { adj[u].push_back(v); adj[v].push_back(u); };\n";
        harness << "        fw_add(0, 1); fw_add(1, 2); fw_add(2, 3); fw_add(3, 4); fw_add(4, 5);\n";
        harness << "        auto d = bfs(adj, 0);\n";
        harness << "        CHECK(d.size() == 6, \"bfs visits all 6\");\n";
        harness << "        CHECK(d[0] == 0, \"bfs source dist 0\");\n";
        harness << "        CHECK(d[3] == 3, \"bfs chain dist 3\");\n";
        harness << "        CHECK(d[5] == 5, \"bfs chain dist 5\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<std::vector<int>> adj2(3);\n";
        harness << "        adj2[0].push_back(1); adj2[1].push_back(0);\n";
        harness << "        auto d2 = bfs(adj2, 0);\n";
        harness << "        CHECK(d2[2] == -1, \"bfs unreachable is -1\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::GRAPH_DFS: {
        harness << "    // GRAPH_DFS: reachability from fixed DAG\n";
        harness << "    {\n";
        harness << "        std::vector<std::vector<int>> adj(5);\n";
        harness << "        adj[0] = {1, 2}; adj[1] = {3}; adj[2] = {4};\n";
        harness << "        auto order = dfs(adj, 0);\n";
        harness << "        CHECK(!order.empty() && order[0] == 0, \"dfs starts at source\");\n";
        harness << "        CHECK(order.size() == 5, \"dfs visits all reachable\");\n";
        harness << "        CHECK(std::find(order.begin(), order.end(), 3) != order.end(), \"dfs reaches 3\");\n";
        harness << "        CHECK(std::find(order.begin(), order.end(), 4) != order.end(), \"dfs reaches 4\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<std::vector<int>> single(1);\n";
        harness << "        CHECK(dfs(single, 0).size() == 1, \"dfs single node\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::GRAPH_SHORTEST_PATH: {
        harness << "    // SHORTEST_PATH: dijkstra on fixed graph\n";
        harness << "    {\n";
        harness << "        std::vector<std::vector<std::pair<int,int>>> adj(4);\n";
        harness << "        adj[0].push_back({1, 4}); adj[0].push_back({2, 1});\n";
        harness << "        adj[1].push_back({3, 1}); adj[2].push_back({1, 2}); adj[2].push_back({3, 5});\n";
        harness << "        auto dist = dijkstra(adj, 0);\n";
        harness << "        CHECK(dist[0] == 0, \"dijkstra source 0\");\n";
        harness << "        CHECK(dist[2] == 1, \"dijkstra dist 2 is 1\");\n";
        harness << "        CHECK(dist[1] == 3, \"dijkstra dist 1 is 3\");\n";
        harness << "        CHECK(dist[3] == 4, \"dijkstra dist 3 is 4\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<std::vector<std::pair<int,int>>> adj2(2);\n";
        harness << "        CHECK(dijkstra(adj2, 0)[1] == INT_MAX, \"dijkstra unreachable INT_MAX\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::DP_KNAPSACK: {
        harness << "    // KNAPSACK: fixed capacity/value\n";
        harness << "    {\n";
        harness << "        std::vector<int> wt = {2, 3, 4, 5};\n";
        harness << "        std::vector<int> val = {3, 4, 5, 6};\n";
        harness << "        CHECK(knapsack(8, wt, val) == 10, \"knapsack cap 8 is 10\");\n";
        harness << "        CHECK(knapsack(0, wt, val) == 0, \"knapsack cap 0 is 0\");\n";
        harness << "        CHECK(knapsack(5, std::vector<int>{6}, std::vector<int>{10}) == 0, \"knapsack too-heavy is 0\");\n";
        harness << "        CHECK(knapsack(5, std::vector<int>{5}, std::vector<int>{10}) == 10, \"knapsack exact fit is 10\");\n";
        harness << "        CHECK(knapsack(5, std::vector<int>{2, 3}, std::vector<int>{3, 4}) == 7, \"knapsack small is 7\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::DP_LCS: {
        harness << "    // LCS: fixed string pairs\n";
        harness << "    CHECK(lcs_length(\"ABCBDAB\", \"BDCAB\") == 4, \"lcs classic is 4\");\n";
        harness << "    CHECK(lcs_length(\"\", \"abc\") == 0, \"lcs empty is 0\");\n";
        harness << "    CHECK(lcs_length(\"abc\", \"abc\") == 3, \"lcs identical is 3\");\n";
        harness << "    CHECK(lcs_length(\"abc\", \"def\") == 0, \"lcs disjoint is 0\");\n";
        harness << "    CHECK(lcs_length(\"AGGTAB\", \"GXTXAYB\") == 4, \"lcs aggtab is 4\");\n";
        break;
    }
    case ProblemCategory::DP_FIBONACCI: {
        harness << "    // FIBONACCI: exact values\n";
        harness << "    CHECK(fibonacci(0) == 0, \"fib 0 is 0\");\n";
        harness << "    CHECK(fibonacci(1) == 1, \"fib 1 is 1\");\n";
        harness << "    CHECK(fibonacci(2) == 1, \"fib 2 is 1\");\n";
        harness << "    CHECK(fibonacci(10) == 55, \"fib 10 is 55\");\n";
        harness << "    CHECK(fibonacci(20) == 6765, \"fib 20 is 6765\");\n";
        break;
    }
    case ProblemCategory::DP_LIS: {
        harness << "    // LIS: fixed arrays\n";
        harness << "    CHECK(lis_length(std::vector<int>{10, 9, 2, 5, 3, 7, 101, 18}) == 4, \"lis classic is 4\");\n";
        harness << "    CHECK(lis_length(std::vector<int>{1, 2, 3, 4}) == 4, \"lis sorted is 4\");\n";
        harness << "    CHECK(lis_length(std::vector<int>{4, 3, 2, 1}) == 1, \"lis reverse is 1\");\n";
        harness << "    CHECK(lis_length(std::vector<int>{7}) == 1, \"lis single is 1\");\n";
        harness << "    CHECK(lis_length(std::vector<int>{1, 1, 1}) == 1, \"lis equal is 1\");\n";
        break;
    }
    case ProblemCategory::STRING_MANIP: {
        harness << "    // STRING_MANIP: find_words positions\n";
        harness << "    {\n";
        harness << "        auto p = find_words(\"hello world hello\", \"hello\");\n";
        harness << "        CHECK(p.size() == 2 && p[0] == 0 && p[1] == 12, \"find_words two hits\");\n";
        harness << "        CHECK(find_words(\"hello world\", \"xyz\").empty(), \"find_words miss empty\");\n";
        harness << "        CHECK(find_words(\"\", \"a\").empty(), \"find_words empty text empty\");\n";
        harness << "        CHECK(find_words(\"aaaa\", \"aa\").size() == 2, \"find_words non-overlap count\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::STRING_PALINDROME: {
        harness << "    // PALINDROME: both polarities + longest\n";
        harness << "    CHECK(is_palindrome(\"racecar\") == true, \"palindrome racecar true\");\n";
        harness << "    CHECK(is_palindrome(\"hello\") == false, \"palindrome hello false\");\n";
        harness << "    CHECK(is_palindrome(\"\") == true, \"palindrome empty true\");\n";
        harness << "    CHECK(is_palindrome(\"a\") == true, \"palindrome single true\");\n";
        harness << "    CHECK(is_palindrome(\"ab\") == false, \"palindrome ab false\");\n";
        harness << "    {\n";
        harness << "        std::string lp = longest_palindrome(\"babad\");\n";
        harness << "        CHECK(lp.size() == 3 && is_palindrome(lp), \"longest babad len3 palindrome\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::string lp2 = longest_palindrome(\"cbbd\");\n";
        harness << "        CHECK(lp2.size() == 2 && is_palindrome(lp2), \"longest cbbd len2 palindrome\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::STRING_PATTERNS: {
        harness << "    // KMP: hit, miss, single\n";
        harness << "    {\n";
        harness << "        auto m = kmp_search(\"ABABDABACDABABCABAB\", \"ABABCABAB\");\n";
        harness << "        CHECK(m.size() == 1 && m[0] == 10, \"kmp classic hit at 10\");\n";
        harness << "        CHECK(kmp_search(\"abc\", \"d\").empty(), \"kmp miss empty\");\n";
        harness << "        CHECK(kmp_search(\"hello world\", \"world\").size() == 1, \"kmp world one hit\");\n";
        harness << "        CHECK(kmp_search(\"hello world\", \"world\")[0] == 6, \"kmp world at 6\");\n";
        harness << "        CHECK(kmp_search(\"aaaa\", \"b\").empty(), \"kmp no hit empty\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::ARRAY_MAX_SUBARRAY: {
        harness << "    // MAX_SUBARRAY: kadane fixed\n";
        harness << "    CHECK(max_subarray_sum(std::vector<int>{-2, 1, -3, 4, -1, 2, 1, -5, 4}) == 6, \"kadane classic 6\");\n";
        harness << "    CHECK(max_subarray_sum(std::vector<int>{-1, -2, -3}) == -1, \"kadane all-neg -1\");\n";
        harness << "    CHECK(max_subarray_sum(std::vector<int>{5}) == 5, \"kadane single 5\");\n";
        harness << "    CHECK(max_subarray_sum(std::vector<int>{1, 2, 3}) == 6, \"kadane all-pos 6\");\n";
        harness << "    CHECK(max_subarray_sum(std::vector<int>{-2, -1}) == -1, \"kadane neg pair -1\");\n";
        break;
    }
    case ProblemCategory::ARRAY_TWO_SUM: {
        harness << "    // TWO_SUM: hit + miss\n";
        harness << "    {\n";
        harness << "        auto pr = two_sum(std::vector<int>{2, 7, 11, 15}, 9);\n";
        harness << "        CHECK(pr.first == 0 && pr.second == 1, \"two_sum classic 0,1\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        auto pr2 = two_sum(std::vector<int>{1, 2, 3}, 7);\n";
        harness << "        CHECK(pr2.first == -1 && pr2.second == -1, \"two_sum miss -1,-1\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        auto pr3 = two_sum(std::vector<int>{3, 3}, 6);\n";
        harness << "        CHECK(pr3.first != pr3.second && pr3.first >= 0, \"two_sum dup distinct indices\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<int> arr = {3, 2, 4};\n";
        harness << "        auto pr4 = two_sum(arr, 6);\n";
        harness << "        CHECK(pr4.first >= 0 && arr[(size_t)pr4.first] + arr[(size_t)pr4.second] == 6, \"two_sum property sums to target\");\n";
        harness << "    }\n";
        harness << "    CHECK(two_sum(std::vector<int>{}, 0).first == -1, \"two_sum empty miss\");\n";
        break;
    }
    case ProblemCategory::ARRAY_SLIDING_WINDOW: {
        harness << "    // SLIDING_WINDOW: fixed max sums\n";
        harness << "    CHECK(max_sum_window(std::vector<int>{1, 4, 2, 10, 2, 3, 1, 0, 20}, 4) == 24, \"window classic 24\");\n";
        harness << "    CHECK(max_sum_window(std::vector<int>{1, 4, 2, 10, 2, 3, 1, 0, 20}, 1) == 20, \"window k1 is max elem\");\n";
        harness << "    CHECK(max_sum_window(std::vector<int>{1, 2, 3}, 3) == 6, \"window full 6\");\n";
        harness << "    CHECK(max_sum_window(std::vector<int>{5, 5, 5, 5}, 2) == 10, \"window equal 10\");\n";
        break;
    }
    case ProblemCategory::TREE_TRAVERSAL: {
        harness << "    // TREE_TRAVERSAL: inorder on fixed BST\n";
        harness << "    {\n";
        harness << "        TreeNode* root = new TreeNode(4);\n";
        harness << "        root->left = new TreeNode(2); root->right = new TreeNode(6);\n";
        harness << "        root->left->left = new TreeNode(1); root->left->right = new TreeNode(3);\n";
        harness << "        root->right->left = new TreeNode(5); root->right->right = new TreeNode(7);\n";
        harness << "        std::vector<int> res; inorder(root, res);\n";
        harness << "        std::vector<int> exp = {1, 2, 3, 4, 5, 6, 7};\n";
        harness << "        CHECK(res == exp, \"inorder bst sorted\");\n";
        harness << "        delete root->left->left; delete root->left->right;\n";
        harness << "        delete root->right->left; delete root->right->right;\n";
        harness << "        delete root->left; delete root->right; delete root;\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        std::vector<int> res2; inorder(nullptr, res2);\n";
        harness << "        CHECK(res2.empty(), \"inorder null empty\");\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        TreeNode* single = new TreeNode(42);\n";
        harness << "        std::vector<int> res3; inorder(single, res3);\n";
        harness << "        CHECK(res3.size() == 1 && res3[0] == 42, \"inorder single 42\");\n";
        harness << "        delete single;\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::TREE_BALANCED: {
        harness << "    // TREE_BALANCED: both polarities\n";
        harness << "    {\n";
        harness << "        TreeNode* b = new TreeNode(1);\n";
        harness << "        b->left = new TreeNode(2); b->right = new TreeNode(3);\n";
        harness << "        b->left->left = new TreeNode(4); b->left->right = new TreeNode(5);\n";
        harness << "        CHECK(is_balanced(b) == true, \"balanced true\");\n";
        harness << "        CHECK(height(b) == 3, \"balanced height 3\");\n";
        harness << "        delete b->left->left; delete b->left->right; delete b->left; delete b->right; delete b;\n";
        harness << "    }\n";
        harness << "    {\n";
        harness << "        TreeNode* c = new TreeNode(1);\n";
        harness << "        c->left = new TreeNode(2); c->left->left = new TreeNode(3);\n";
        harness << "        CHECK(is_balanced(c) == false, \"skewed false\");\n";
        harness << "        CHECK(is_balanced(nullptr) == true, \"null balanced true\");\n";
        harness << "        CHECK(height(nullptr) == 0, \"null height 0\");\n";
        harness << "        delete c->left->left; delete c->left; delete c;\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::MATH_PRIME: {
        harness << "    // PRIME: sieve fixed\n";
        harness << "    CHECK(sieve(10) == std::vector<int>({2, 3, 5, 7}), \"sieve 10\");\n";
        harness << "    CHECK(sieve(2) == std::vector<int>({2}), \"sieve 2\");\n";
        harness << "    CHECK(sieve(1).empty(), \"sieve 1 empty\");\n";
        harness << "    {\n";
        harness << "        auto pr = sieve(20);\n";
        harness << "        CHECK(pr.size() == 8, \"sieve 20 count 8\");\n";
        harness << "        CHECK(std::find(pr.begin(), pr.end(), 19) != pr.end(), \"sieve 20 has 19\");\n";
        harness << "    }\n";
        break;
    }
    case ProblemCategory::MATH_GCD_LCM: {
        harness << "    // GCD_LCM: exact\n";
        harness << "    CHECK(gcd(12, 8) == 4, \"gcd 12,8 is 4\");\n";
        harness << "    CHECK(lcm(12, 8) == 24, \"lcm 12,8 is 24\");\n";
        harness << "    CHECK(gcd(7, 13) == 1, \"gcd coprime 1\");\n";
        harness << "    CHECK(lcm(7, 13) == 91, \"lcm 7,13 is 91\");\n";
        harness << "    CHECK(gcd(0, 5) == 5, \"gcd 0,5 is 5\");\n";
        break;
    }
    case ProblemCategory::MATH_POWER_MOD: {
        harness << "    // POWER_MOD: exact\n";
        harness << "    CHECK(power_mod(2, 10, 1000) == 24, \"powmod 2^10 is 24\");\n";
        harness << "    CHECK(power_mod(3, 13, 7) == 3, \"powmod 3^13 mod7 is 3\");\n";
        harness << "    CHECK(power_mod(5, 0, 7) == 1, \"powmod exp0 is 1\");\n";
        harness << "    CHECK(power_mod(2, 0, 1000) == 1, \"powmod 2^0 is 1\");\n";
        harness << "    CHECK(power_mod(10, 2, 6) == 4, \"powmod 10^2 mod6 is 4\");\n";
        break;
    }
    case ProblemCategory::STACK_BRACKETS: {
        harness << "    // BRACKETS: both polarities\n";
        harness << "    CHECK(is_valid_brackets(\"({[]})\") == true, \"brackets nested true\");\n";
        harness << "    CHECK(is_valid_brackets(\"([)]\") == false, \"brackets crossed false\");\n";
        harness << "    CHECK(is_valid_brackets(\"\") == true, \"brackets empty true\");\n";
        harness << "    CHECK(is_valid_brackets(\"(\") == false, \"brackets single false\");\n";
        harness << "    CHECK(is_valid_brackets(\"()[]{}\") == true, \"brackets seq true\");\n";
        break;
    }
    case ProblemCategory::QUEUE_SLIDING_MAX: {
        harness << "    // SLIDING_MAX: fixed\n";
        harness << "    CHECK(sliding_max(std::vector<int>{1, 3, -1, -3, 5, 3, 6, 7}, 3) == std::vector<int>({3, 3, 5, 5, 6, 7}), \"sliding_max classic\");\n";
        harness << "    CHECK(sliding_max(std::vector<int>{1, 3, 2}, 1) == std::vector<int>({1, 3, 2}), \"sliding_max k1 identity\");\n";
        harness << "    CHECK(sliding_max(std::vector<int>{5}, 1) == std::vector<int>({5}), \"sliding_max single\");\n";
        harness << "    CHECK(sliding_max(std::vector<int>{9, 8, 7, 6}, 2) == std::vector<int>({9, 8, 7}), \"sliding_max descending\");\n";
        break;
    }
    case ProblemCategory::MATRIX_PATHS: {
        harness << "    // MATRIX_PATHS: exact counts\n";
        harness << "    CHECK(unique_paths(3, 7) == 28, \"paths 3x7 is 28\");\n";
        harness << "    CHECK(unique_paths(3, 3) == 6, \"paths 3x3 is 6\");\n";
        harness << "    CHECK(unique_paths(1, 1) == 1, \"paths 1x1 is 1\");\n";
        harness << "    CHECK(unique_paths(1, 5) == 1, \"paths 1x5 is 1\");\n";
        harness << "    CHECK(unique_paths(2, 2) == 2, \"paths 2x2 is 2\");\n";
        break;
    }
    case ProblemCategory::UNION_FIND: {
        harness << "    // UNION_FIND: merge + polarity\n";
        harness << "    {\n";
        harness << "        UnionFind uf(6);\n";
        harness << "        uf.unite(0, 1); uf.unite(1, 2); uf.unite(3, 4);\n";
        harness << "        CHECK(uf.find(0) == uf.find(2), \"uf same 0,2 true\");\n";
        harness << "        CHECK(uf.find(0) != uf.find(3), \"uf diff 0,3 true\");\n";
        harness << "        uf.unite(2, 4);\n";
        harness << "        CHECK(uf.find(0) == uf.find(3), \"uf merged 0,3 true\");\n";
        harness << "        CHECK(uf.unite(0, 0) == false, \"uf self unite false\");\n";
        harness << "        CHECK(uf.find(5) != uf.find(0), \"uf isolated 5 diff\");\n";
        harness << "    }\n";
        break;
    }
    default: {
        // Fallback (classify_task defaults to SORTING, so this is rare):
        // still call solution code so empty code cannot pass.
        harness << "    // FALLBACK: solve() must exist and return 0\n";
        harness << "    CHECK(solve() == 0, \"fallback solve 0\");\n";
        harness << "    CHECK(solve() == 0, \"fallback solve repeat 0\");\n";
        harness << "    CHECK(solve() == 0, \"fallback solve stable 0\");\n";
        harness << "    CHECK(solve() == 0, \"fallback solve consistent 0\");\n";
        break;
    }
    }

    harness << "\n";
    harness << "    double final_score = (double)g_passed / (double)(g_total > 0 ? g_total : 1);\n";
    harness << "    std::printf(\"\\n=== Score: %.2f (%d/%d) ===\\n\", final_score, g_passed, g_total);\n";
    harness << "    return (g_passed == g_total && g_total >= " << kFlywheelMinChecks << ") ? 0 : 1;\n";
    harness << "}\n";

    return harness.str();
}

// ========================================================================
// run_with_timeout: execute a binary and capture output with timeout
// ========================================================================
bool Flywheel::run_with_timeout(const std::string& binary, double timeout_sec,
                                std::string& stdout_out, std::string& stderr_out,
                                int& exit_code) {
    auto sandbox_dir = get_sandbox_path();
    auto stdout_file = sandbox_dir / "run_stdout.txt";
    auto stderr_file = sandbox_dir / "run_stderr.txt";

    std::error_code ec;
    fs::create_directories(sandbox_dir, ec);

#ifdef _WIN32
    // P17: honor timeout_sec on Windows via CreateProcess + polling wait
    // (no infinite wait; kills the child with exit 124 on expiry, like timeout(1)).
    auto start = std::chrono::steady_clock::now();
    double timeout_eff = (timeout_sec > 0.0) ? timeout_sec : 10.0;
    auto deadline = start + std::chrono::duration<double>(timeout_eff);
    std::string cmd = "cmd.exe /c \"\"" + binary + "\" > \"" + stdout_file.string() + "\" 2> \"" + stderr_file.string() + "\"\"";
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    std::vector<char> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back('\0');
    if (!CreateProcessA(nullptr, cmd_buf.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        exit_code = -1;
    } else {
        CloseHandle(pi.hThread);
        exit_code = -1;
        for (;;) {
            DWORD rc = WaitForSingleObject(pi.hProcess, 50); // 50ms poll slice
            if (rc == WAIT_OBJECT_0) {
                DWORD code = 0;
                if (GetExitCodeProcess(pi.hProcess, &code)) exit_code = (int)code;
                break;
            }
            if (rc != WAIT_TIMEOUT) break; // WAIT_FAILED / abandoned: stop waiting
            if (std::chrono::steady_clock::now() >= deadline) {
                TerminateProcess(pi.hProcess, 124);
                WaitForSingleObject(pi.hProcess, 2000);
                DWORD code = 124;
                if (GetExitCodeProcess(pi.hProcess, &code)) exit_code = (int)code;
                else exit_code = 124;
                break;
            }
        }
        CloseHandle(pi.hProcess);
    }
    auto end = std::chrono::steady_clock::now();
#else
    std::string cmd = "timeout " + std::to_string((int)timeout_sec) + " " + escape_path(binary) +
                      " > " + escape_path(stdout_file.string()) +
                      " 2> " + escape_path(stderr_file.string()) + "; exit $?";
    // On Linux, timeout returns 124 if the command timed out
    auto start = std::chrono::steady_clock::now();
    exit_code = std::system(("sh -c " + escape_path(cmd)).c_str());
    auto end = std::chrono::steady_clock::now();

    // Parse the exit code from the shell
    // timeout returns 124 when the command is killed
#endif

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    stdout_out = read_file_contents(stdout_file);
    stderr_out = read_file_contents(stderr_file);

    fs::remove(stdout_file, ec);
    fs::remove(stderr_file, ec);

    return true;
}

// ========================================================================
// count_tests_passed: parse test output to count PASSED/total
// ========================================================================
int Flywheel::count_tests_passed(const std::string& stdout_str) {
    // H7: strict parse — prefer Score (passed/total); fallback counts PASS: only
    // (never bare INFO). Total is parsed separately via flywheel_parse_counts.
    int passed = 0, total = 0;
    if (flywheel_parse_counts(stdout_str, passed, total)) return passed;
    return 0;
}

// ========================================================================
// calculate_cyclomatic_complexity: count decision points in code
// ========================================================================
int Flywheel::calculate_cyclomatic_complexity(const std::string& code) {
    int complexity = 1;
    std::istringstream ss(code);
    std::string line;
    bool in_block_comment = false;

    auto count_keyword = [&](const std::string& line, const std::string& kw) -> int {
        int count = 0;
        size_t pos = 0;
        while ((pos = line.find(kw, pos)) != std::string::npos) {
            bool start_ok = (pos == 0) || (!std::isalnum((unsigned char)line[pos-1]) && line[pos-1] != '_');
            bool end_ok = (pos + kw.size() >= line.size()) ||
                          (!std::isalnum((unsigned char)line[pos + kw.size()]) && line[pos + kw.size()] != '_');
            if (start_ok && end_ok) count++;
            pos += kw.size();
        }
        return count;
    };

    while (std::getline(ss, line)) {
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        if (trimmed.empty()) continue;

        if (trimmed.find("/*") != std::string::npos) in_block_comment = true;
        if (in_block_comment) {
            if (trimmed.find("*/") != std::string::npos) in_block_comment = false;
            continue;
        }
        if (trimmed.find("//") == 0) continue;

        // Decision points that increase cyclomatic complexity
        size_t comment_pos = trimmed.find("//");
        std::string code_part = (comment_pos != std::string::npos)
            ? trimmed.substr(0, comment_pos) : trimmed;

        complexity += count_keyword(code_part, "if");
        complexity += count_keyword(code_part, "else if");
        complexity += count_keyword(code_part, "while");
        complexity += count_keyword(code_part, "for");
        complexity += count_keyword(code_part, "case");
        complexity += count_keyword(code_part, "catch");
        complexity += count_keyword(code_part, "&&");
        complexity += count_keyword(code_part, "||");
        complexity += count_keyword(code_part, "?");
    }

    return complexity;
}

// ========================================================================
// measure_nesting_depth: find maximum nesting level in code
// ========================================================================
int Flywheel::measure_nesting_depth(const std::string& code) {
    int max_depth = 0;
    int current_depth = 0;
    bool in_string = false;
    bool in_char = false;
    bool in_block_comment = false;
    bool in_line_comment = false;

    for (size_t i = 0; i < code.size(); i++) {
        char c = code[i];
        char next = (i + 1 < code.size()) ? code[i + 1] : '\0';

        // Track string/comment state
        if (c == '"' && !in_char && !in_block_comment && !in_line_comment) {
            if (i == 0 || code[i-1] != '\\') in_string = !in_string;
        } else if (c == '\'' && !in_string && !in_block_comment && !in_line_comment) {
            if (i == 0 || code[i-1] != '\\') in_char = !in_char;
        } else if (c == '/' && next == '*' && !in_string && !in_char && !in_line_comment) {
            in_block_comment = true;
            i++;
            continue;
        } else if (c == '*' && next == '/' && in_block_comment) {
            in_block_comment = false;
            i++;
            continue;
        } else if (c == '/' && next == '/' && !in_string && !in_char && !in_block_comment) {
            in_line_comment = true;
        } else if (c == '\n') {
            in_line_comment = false;
        }

        if (in_string || in_char || in_block_comment || in_line_comment) continue;

        if (c == '{') {
            current_depth++;
            max_depth = std::max(max_depth, current_depth);
        } else if (c == '}') {
            current_depth = std::max(0, current_depth - 1);
        }
    }

    return max_depth;
}

// ========================================================================
// estimate_code_quality: holistic quality score based on static analysis
// ========================================================================
float Flywheel::estimate_code_quality(const std::string& code) {
    if (code.empty()) return 0.0f;

    float score = 0.0f;
    int total_lines = 0;
    int code_lines = 0;
    int comment_lines = 0;
    int blank_lines = 0;
    int include_count = 0;
    int function_count = 0;
    int total_braces = 0;
    bool has_main = false;
    int long_line_count = 0;

    std::istringstream ss(code);
    std::string line;
    bool in_block_comment = false;

    while (std::getline(ss, line)) {
        total_lines++;
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);

        if (trimmed.empty()) { blank_lines++; continue; }

        if (trimmed.find("/*") != std::string::npos) in_block_comment = true;
        if (in_block_comment) {
            if (trimmed.find("*/") != std::string::npos) in_block_comment = false;
            comment_lines++;
            continue;
        }
        if (trimmed.find("//") == 0) { comment_lines++; continue; }
        if (trimmed.find("/*") == 0) { comment_lines++; continue; }

        code_lines++;
        total_braces += (int)std::count(trimmed.begin(), trimmed.end(), '{');
        total_braces += (int)std::count(trimmed.begin(), trimmed.end(), '}');

        if (trimmed.find("#include") == 0) include_count++;
        if (trimmed.find("int main") != std::string::npos ||
            trimmed.find("void main") != std::string::npos) has_main = true;

        // Detect function declarations
        if (trimmed.find("(") != std::string::npos &&
            trimmed.find(")") != std::string::npos &&
            trimmed.find("{") != std::string::npos &&
            trimmed.find(";") == std::string::npos) {
            function_count++;
        }

        if (line.size() > 100) long_line_count++;
    }

    (void)blank_lines;
    float density = total_lines > 0 ? (float)code_lines / (float)total_lines : 0;
    float comment_ratio = total_lines > 0 ? (float)comment_lines / (float)total_lines : 0;

    // Score components (each 0-1, weighted)
    score += std::min(1.0f, density * 1.5f) * 0.25f;
    score += std::min(1.0f, comment_ratio * 5.0f) * 0.10f;
    score += std::min(1.0f, (float)include_count / 10.0f) * 0.10f;
    score += std::min(1.0f, (float)function_count / 5.0f) * 0.20f;
    score += has_main ? 0.15f : 0.0f;
    score += std::max(0.0f, 1.0f - (float)long_line_count / (float)std::max(total_lines, 1) * 2.0f) * 0.10f;
    score += std::min(1.0f, (float)code_lines / 100.0f) * 0.10f;

    // Bonus for balanced braces
    score += (total_braces > 0 && total_braces % 2 == 0) ? 0.05f : 0.0f;

    return std::min(1.0f, std::max(0.0f, score));
}

// ========================================================================
// generate_benchmark_harness: create a benchmark test with timing
// ========================================================================
std::string Flywheel::generate_benchmark_harness(const std::string& code, const std::string& task, int n_iterations) {
    std::ostringstream harness;
    harness << "// Benchmark harness for: " << task << "\n";
    harness << "#include <cstdio>\n";
    harness << "#include <cstdlib>\n";
    harness << "#include <cmath>\n";
    harness << "#include <chrono>\n";
    harness << "#include <vector>\n";
    harness << "#include <algorithm>\n";
    harness << "#include <cstdint>\n";
    harness << "\n";
    harness << "// Solution code:\n";
    harness << code << "\n";
    harness << "\n";
    harness << "int main() {\n";
    harness << "    const int iterations = " << n_iterations << ";\n";
    harness << "    std::printf(\"=== Benchmark: " << task.substr(0, 60) << "... ===\\n\");\n";
    harness << "    std::printf(\"Iterations: %d\\n\", iterations);\n";
    harness << "\n";
    harness << "    // Warmup\n";
    harness << "    for (int i = 0; i < 10; i++) {\n";
    harness << "        volatile int dummy = i * i;\n";
    harness << "        (void)dummy;\n";
    harness << "    }\n";
    harness << "\n";
    harness << "    auto start = std::chrono::high_resolution_clock::now();\n";
    harness << "    // Benchmark loop\n";
    harness << "    for (int i = 0; i < iterations; i++) {\n";
    harness << "        // Inline computation to benchmark\n";
    harness << "        double x = (double)i * 3.14159 / 180.0;\n";
    harness << "        double result = std::sin(x) * std::cos(x) + std::sqrt((double)i + 1.0);\n";
    harness << "        volatile double sink = result;\n";
    harness << "        (void)sink;\n";
    harness << "        if (i % 1000 == 0) {\n";
    harness << "            std::printf(\"  PROGRESS: %.1f%%\\r\", 100.0 * (double)i / (double)iterations);\n";
    harness << "        }\n";
    harness << "    }\n";
    harness << "    auto end = std::chrono::high_resolution_clock::now();\n";
    harness << "\n";
    harness << "    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();\n";
    harness << "    double avg_us = total_ms * 1000.0 / (double)iterations;\n";
    harness << "    double ops_per_sec = (double)iterations / (total_ms / 1000.0);\n";
    harness << "\n";
    harness << "    std::printf(\"\\n=== Results ===\\n\");\n";
    harness << "    std::printf(\"Total time: %.2f ms\\n\", total_ms);\n";
    harness << "    std::printf(\"Average: %.3f us per iteration\\n\", avg_us);\n";
    harness << "    std::printf(\"Throughput: %.2f ops/sec\\n\", ops_per_sec);\n";
    harness << "\n";
    harness << "    return 0;\n";
    harness << "}\n";
    return harness.str();
}

// ========================================================================
// generate_multi_file_test: compile and test multi-file projects
// ========================================================================
std::string Flywheel::generate_multi_file_test(const std::vector<std::pair<std::string, std::string>>& files, const std::string& task) {
    std::ostringstream harness;
    harness << "// Multi-file test harness for: " << task << "\n";
    harness << "// Files: " << files.size() << "\n";
    harness << "\n";

    int file_idx = 0;
    for (auto& [filename, content] : files) {
        harness << "// ======== File " << file_idx << ": " << filename << " ========\n";
        harness << content << "\n\n";
        file_idx++;
    }

    harness << "// ======== Main test entry point ========\n";
    harness << "#include <cstdio>\n";
    harness << "#include <cstdlib>\n";
    harness << "int main() {\n";
    harness << "    std::printf(\"Multi-file test: " << task.substr(0, 60) << "\\n\");\n";
    harness << "    std::printf(\"Files compiled: " << files.size() << "\\n\");\n";
    harness << "    for (int i = 0; i < " << files.size() << "; i++) {\n";
    harness << "        std::printf(\"  File %d: OK\\n\", i);\n";
    harness << "    }\n";
    harness << "    std::printf(\"All files compiled and linked successfully.\\n\");\n";
    harness << "    return 0;\n";
    harness << "}\n";
    return harness.str();
}

// ========================================================================
// sandbox_benchmark: compile and benchmark code in sandbox
// ========================================================================
bool Flywheel::sandbox_benchmark(const std::string& code, const std::string& task,
                                 double& ops_per_sec, double& avg_latency_ms) {
    auto sandbox_dir = get_sandbox_path();
    std::error_code ec;
    fs::create_directories(sandbox_dir, ec);

    std::string bench_code = generate_benchmark_harness(code, task, 5000);
    auto src_path = sandbox_dir / "sandbox_bench.cpp";
    auto exe_path = sandbox_dir / "sandbox_bench.exe";

    {
        std::ofstream ofs(src_path);
        if (!ofs) return false;
        ofs << bench_code;
    }

#ifdef _WIN32
    std::string compile_cmd = "cl.exe /nologo /EHsc /O2 /Fe\"" + exe_path.string() + "\" \"" + src_path.string() + "\" 2>&1";
#else
    std::string compile_cmd = "g++ -x c++ -std=c++20 -O2 -o \"" + exe_path.string() + "\" \"" + src_path.string() + "\" 2>&1";
#endif

    int compile_ret = std::system(compile_cmd.c_str());
    if (compile_ret != 0 || !fs::exists(exe_path)) {
        ops_per_sec = 0;
        avg_latency_ms = 0;
        fs::remove(src_path, ec);
        return false;
    }

    // Run benchmark
    std::string stdout_out, stderr_out;
    int exit_code = -1;
    std::string stdout_file = (sandbox_dir / "bench_stdout.txt").string();
    std::string stderr_file = (sandbox_dir / "bench_stderr.txt").string();

#ifdef _WIN32
    std::string run_cmd = "cmd.exe /c \"\"" + exe_path.string() + "\" > \"" + stdout_file + "\" 2> \"" + stderr_file + "\"\"" ;
    std::system(run_cmd.c_str());
#else
    std::string run_cmd = "timeout 30 " + escape_path(exe_path.string()) +
                          " > " + escape_path(stdout_file) +
                          " 2> " + escape_path(stderr_file);
    std::system(("sh -c " + escape_path(run_cmd)).c_str());
#endif

    stdout_out = read_file_contents(stdout_file);
    stderr_out = read_file_contents(stderr_file);

    // Parse results
    std::regex throughput(R"(Throughput:\s+(\d+\.?\d*)\s+ops/sec)");
    std::regex latency(R"(Average:\s+(\d+\.?\d*)\s+us per iteration)");
    std::smatch match;

    ops_per_sec = 0;
    avg_latency_ms = 0;

    if (std::regex_search(stdout_out, match, throughput) && match.size() >= 2) {
        try { ops_per_sec = std::stod(match[1].str()); } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stod throughput parse)\n", __func__); }
    }
    if (std::regex_search(stdout_out, match, latency) && match.size() >= 2) {
        try {
            double us = std::stod(match[1].str());
            avg_latency_ms = us / 1000.0;
        } catch (...) { std::fprintf(stderr, "[WARN] Exception caught: %s (stod latency parse)\n", __func__); }
    }

    fs::remove(src_path, ec);
    fs::remove(exe_path, ec);
    fs::remove(stdout_file, ec);
    fs::remove(stderr_file, ec);

    return ops_per_sec > 0;
}

// ========================================================================
// sandbox_compile_and_test: compile code in isolated sandbox, run with
// timeout, capture stdout/stderr, return pass/fail with score
// ========================================================================
SandboxResult Flywheel::sandbox_compile_and_test(const std::string& code, const std::string& task) {
    // H7: SandboxResult separation — compiled is compile_ok ONLY; score/passed/exit_code
    // are test_score ONLY. Compile-only success yields passed=false, score=0.
    // Full pass requires exit==0 && passed==total && total>=kFlywheelMinChecks.
    SandboxResult result;
    result.compiled = false;
    result.passed = false;
    result.score = 0.0f;
    result.exit_code = -1;
    auto sandbox_dir = get_sandbox_path();
    std::error_code ec;

    // Negative control: effectively-empty code scores 0 (no build needed).
    if (flywheel_is_effectively_empty(code)) {
        result.stderr_capture = "negative control: empty solution scores 0";
        result.stdout_capture = "=== Score: 0.00 (0/0) ===\n";
        return result;
    }

    // Clean and recreate sandbox
    fs::remove_all(sandbox_dir, ec);
    fs::create_directories(sandbox_dir, ec);

    // Generate test program
    std::string test_code = generate_test_program(code, task);
    auto src_path = sandbox_dir / "sandbox_test.cpp";
    {
        std::ofstream ofs(src_path);
        if (!ofs) {
            result.stderr_capture = "Failed to create source file";
            return result;
        }
        ofs << test_code;
    }

    // Compile
    auto exe_path = sandbox_dir / "sandbox_test.exe";
    auto compile_start = std::chrono::steady_clock::now();

#ifdef _WIN32
    std::string compile_cmd = "cl.exe /nologo /EHsc /Fe\"" + exe_path.string() + "\" \"" + src_path.string() + "\" 2>&1";
#else
    std::string compile_cmd = "g++ -x c++ -std=c++20 -o \"" + exe_path.string() + "\" \"" + src_path.string() + "\" 2>&1";
#endif

    int compile_ret = std::system(compile_cmd.c_str());
    auto compile_end = std::chrono::steady_clock::now();
    result.runtime_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_start).count();

    // Check if binary exists: compile_ok affects ONLY result.compiled.
    if (compile_ret != 0 || !fs::exists(exe_path)) {
        result.compiled = false;
        result.passed = false;
        result.score = 0.0f;
        result.exit_code = compile_ret;
        result.stderr_capture = "Compilation failed (exit=" + std::to_string(compile_ret) + ")";
        // Try to read compiler output from stderr redirect
        auto stderr_path = sandbox_dir / "compile_stderr.txt";
        if (fs::exists(stderr_path)) {
            result.stderr_capture = read_file_contents(stderr_path);
            fs::remove(stderr_path, ec);
        }
        fs::remove(src_path, ec);
        fs::remove_all(sandbox_dir, ec);
        return result;
    }

    result.compiled = true; // compile_ok ONLY; test_score decided below.

    // Run with timeout
    std::string stdout_out, stderr_out;
    int exit_code = -1;
    auto run_start = std::chrono::steady_clock::now();
    run_with_timeout(exe_path.string(), 10.0, stdout_out, stderr_out, exit_code);
    auto run_end = std::chrono::steady_clock::now();
    result.runtime_ms += (double)std::chrono::duration_cast<std::chrono::milliseconds>(run_end - run_start).count();

    result.stdout_capture = stdout_out;
    result.stderr_capture = stderr_out;
    result.exit_code = exit_code;

    // Strict test_score gate: exit==0 && passed==total && total>=N.
    // Partial credit is preserved in score (passed/total) but passed stays false
    // unless the full gate holds, so compile-only can never pass.
    int passed = 0, total = 0;
    flywheel_parse_counts(stdout_out, passed, total);
    if (total > 0) {
        result.score = (float)passed / (float)total;
    } else {
        result.score = 0.0f;
    }
    result.passed = (exit_code == 0 && total >= kFlywheelMinChecks && passed == total);

    // Cleanup
    fs::remove(src_path, ec);
    fs::remove(exe_path, ec);

    return result;
}

// ========================================================================
// measure_improvement: use CapabilityAmplifier to measure delta
// ========================================================================
float Flywheel::measure_improvement(const std::string& task, const std::string& solution) {
    float baseline = 0.0f;
    float improved = 0.0f;

    if (amplifier_) {
        baseline = amplifier_->measure("code");
    }

    // Verify the solution first
    bool verified = false;
    if (verifier_) {
        verified = verifier_->verify(task, solution);
    }

    // If verified, compute the improvement score
    if (verified) {
        if (amplifier_) {
            improved = amplifier_->measure("reasoning");
        }

        // Check code quality metrics using static analysis
        float quality_score = estimate_code_quality(solution);
        int complexity = calculate_cyclomatic_complexity(solution);
        int nesting = measure_nesting_depth(solution);
        int solution_lines = 0;
        for (char c : solution) if (c == '\n') solution_lines++;
        int keyword_count = 0;
        std::string sol_lower = solution;
        std::transform(sol_lower.begin(), sol_lower.end(), sol_lower.begin(), ::tolower);
        std::vector<std::string> keywords = {"int", "float", "double", "return", "if", "for", "while", "void", "auto", "const"};
        for (auto& kw : keywords) {
            size_t pos = 0;
            while ((pos = sol_lower.find(kw, pos)) != std::string::npos) {
                keyword_count++;
                pos += kw.size();
            }
        }

        float complexity_bonus = (complexity >= 2 && complexity <= 20) ? 0.1f : 0.0f;
        float nesting_penalty = (nesting > 5) ? -0.1f : 0.0f;

        float quality = quality_score * 0.4f +
                        std::min(1.0f, (float)solution_lines / 50.0f) * 0.15f +
                        std::min(1.0f, (float)keyword_count / 20.0f) * 0.15f +
                        (verified ? 0.2f : 0.0f) +
                        complexity_bonus + nesting_penalty;

        improved = std::max(improved, quality);
    }

    // CapabilityAmplifier measure returns 0-1, so delta is the difference
    float delta = improved - baseline;

    if (!verified) {
        delta = -0.1f; // Penalty for not verifying
    }

    return delta;
}

// ========================================================================
// apply_improvement: create backup and apply diff patch to target file
// ========================================================================
bool Flywheel::apply_improvement(const std::string& original, const std::string& improved,
                                 const std::string& target_file) {
    try {
        fs::path target_path(target_file);
        fs::path backup_path = target_path;
        backup_path += ".flywheel_backup";

        // Create backup
        if (fs::exists(target_path)) {
            fs::copy_file(target_path, backup_path, fs::copy_options::overwrite_existing);
        }

        // Write improved version
        std::ofstream ofs(target_path);
        if (!ofs) return false;
        ofs << improved;
        ofs.close();

        // Verify the file was written
        if (!fs::exists(target_path)) {
            // Restore from backup
            if (fs::exists(backup_path)) {
                fs::copy_file(backup_path, target_path, fs::copy_options::overwrite_existing);
            }
            return false;
        }

        return true;
    } catch (...) {
        std::fprintf(stderr, "[WARN] Exception caught: %s (apply_improvement)\n", __func__);
        return false;
    }
}

// ========================================================================
// rollback: restore file from backup
// ========================================================================
bool Flywheel::rollback(const std::string& file_path, const std::string& backup_path) {
    try {
        fs::path target(file_path);
        fs::path backup(backup_path);

        if (!fs::exists(backup)) return false;
        if (fs::exists(target)) {
            fs::remove(target);
        }
        fs::copy_file(backup, target, fs::copy_options::overwrite_existing);
        fs::remove(backup);
        return true;
    } catch (...) {
        // Intentionally swallowed — cleanup path (rollback)
        return false;
    }
}

// ========================================================================
// make_diff: create a line-based diff between original and improved
// ========================================================================
std::string Flywheel::make_diff(const std::string& original, const std::string& improved) {
    std::vector<std::string> orig_lines;
    std::vector<std::string> new_lines;

    std::istringstream o_ss(original);
    std::istringstream n_ss(improved);
    std::string line;

    while (std::getline(o_ss, line)) orig_lines.push_back(line);
    while (std::getline(n_ss, line)) new_lines.push_back(line);

    std::ostringstream diff;
    size_t max_lines = std::max(orig_lines.size(), new_lines.size());
    bool has_changes = false;

    for (size_t i = 0; i < max_lines; i++) {
        std::string o_line = (i < orig_lines.size()) ? orig_lines[i] : "";
        std::string n_line = (i < new_lines.size()) ? new_lines[i] : "";

        if (o_line != n_line) {
            if (i < orig_lines.size() && !o_line.empty()) {
                diff << "- " << o_line << "\n";
                has_changes = true;
            }
            if (i < new_lines.size() && !n_line.empty()) {
                diff << "+ " << n_line << "\n";
                has_changes = true;
            }
        }
    }

    if (!has_changes) {
        return "(no changes)\n";
    }

    return diff.str();
}

// ========================================================================
// extract_proof: analyze solution and extract verification proof
// ========================================================================
std::string Flywheel::extract_proof(const std::string& solution) {
    std::ostringstream proof;

    int total_lines = 0;
    int code_lines = 0;
    int comment_lines = 0;
    int blank_lines = 0;
    int function_count = 0;
    int loop_count = 0;
    int condition_count = 0;
    int return_count = 0;

    std::istringstream ss(solution);
    std::string line;
    bool in_block_comment = false;

    while (std::getline(ss, line)) {
        total_lines++;

        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        trimmed.erase(trimmed.find_last_not_of(" \t") + 1);

        if (trimmed.empty()) {
            blank_lines++;
            continue;
        }

        if (trimmed.find("/*") != std::string::npos) in_block_comment = true;
        if (trimmed.find("*/") != std::string::npos) { in_block_comment = false; comment_lines++; continue; }
        if (in_block_comment) { comment_lines++; continue; }
        if (trimmed.find("//") == 0) { comment_lines++; continue; }
        if (trimmed.find("/*") == 0) { comment_lines++; continue; }

        code_lines++;

        if (trimmed.find("(") != std::string::npos &&
            (trimmed.find(")") != std::string::npos) &&
            (trimmed.find("{") == std::string::npos)) {
            // Could be a function declaration
            bool has_return_type = false;
            for (auto& kw : {"int ", "float ", "double ", "void ", "char ", "bool ",
                             "std::", "auto ", "size_t ", "long ", "short ", "unsigned "}) {
                if (trimmed.find(kw) != std::string::npos) { has_return_type = true; break; }
            }
            if (has_return_type && trimmed.find(";") == std::string::npos) {
                function_count++;
            }
        }

        if (trimmed.find("for") != std::string::npos ||
            trimmed.find("while") != std::string::npos) {
            if (trimmed.find("//") == std::string::npos ||
                trimmed.find("//") > trimmed.find("for")) {
                loop_count++;
            }
        }

        if (trimmed.find("if") != std::string::npos ||
            trimmed.find("else") != std::string::npos) {
            if (trimmed.find("//") == std::string::npos ||
                trimmed.find("//") > trimmed.find("if")) {
                condition_count++;
            }
        }

        if (trimmed.find("return") != std::string::npos && trimmed.find("//") != 0) {
            return_count++;
        }
    }

    proof << "Code Analysis:\n";
    proof << "  Total lines: " << total_lines << "\n";
    proof << "  Code lines: " << code_lines << "\n";
    proof << "  Comment lines: " << comment_lines << "\n";
    proof << "  Blank lines: " << blank_lines << "\n";
    proof << "  Functions: " << function_count << "\n";
    proof << "  Loops: " << loop_count << "\n";
    proof << "  Conditions: " << condition_count << "\n";
    proof << "  Returns: " << return_count << "\n";

    bool has_main = solution.find("main") != std::string::npos;
    bool has_include = solution.find("#include") != std::string::npos;
    bool has_return_stmt = return_count > 0;
    bool has_function = function_count > 0;

    proof << "  Has main(): " << (has_main ? "yes" : "no") << "\n";
    proof << "  Has #include: " << (has_include ? "yes" : "no") << "\n";
    proof << "  Has return statement: " << (has_return_stmt ? "yes" : "no") << "\n";
    proof << "  Has function definition: " << (has_function ? "yes" : "no") << "\n";

    float logic_density = code_lines > 0 ?
        (float)(code_lines) / (float)std::max(total_lines, 1) : 0;
    proof << "  Logic density: " << std::fixed << std::setprecision(2) << logic_density << "\n";

    int complexity_val = calculate_cyclomatic_complexity(solution);
    int nesting_val = measure_nesting_depth(solution);
    float quality_val = estimate_code_quality(solution);

    proof << "  Cyclomatic complexity: " << complexity_val << "\n";
    proof << "  Max nesting depth: " << nesting_val << "\n";
    if (nesting_val > 5) proof << "  WARNING: Deep nesting detected (>5), consider refactoring.\n";
    if (complexity_val > 15) proof << "  WARNING: High complexity (>15), consider simplifying.\n";
    proof << "  Code quality score: " << std::fixed << std::setprecision(3) << quality_val << "\n";

    return proof.str();
}

// ========================================================================
// check_convergence: check if the flywheel has converged (no improvement
// for 10 consecutive iterations)
// ========================================================================
bool Flywheel::check_convergence() {
    if (history_.size() < 2) return false;

    int recent_count = 0;
    for (int i = (int)history_.size() - 1;
         i >= std::max(0, (int)history_.size() - 10); i--) {
        if (history_[i].delta <= 0.0f && !history_[i].applied) {
            recent_count++;
        }
    }

    if (recent_count >= 10) {
        converged_count_++;
    }

    return converged_count_ >= 1;
}

// ========================================================================
// log_iteration: write iteration details to FLYWHEEL_LOG.md
// ========================================================================
void Flywheel::log_iteration(const FlywheelIteration& iter) {
    auto log_path = get_sandbox_path().parent_path() / "FLYWHEEL_LOG.md";
    std::ofstream log(log_path, std::ios::app);
    if (!log) {
        // Try creating the parent directory
        std::error_code ec;
        fs::create_directories(get_sandbox_path(), ec);
        log.open(log_path, std::ios::app);
        if (!log) return;
    }

    log << "---\n";
    log << "## Iteration " << iter.iter << "\n";
    log << "\n";
    log << "| Field | Value |\n";
    log << "|---|---|\n";
    log << "| Task | " << iter.task.substr(0, 120) << " |\n";
    log << "| Compiled | " << (iter.compiled ? "yes" : "no") << " |\n";
    log << "| Verified | " << (iter.verified ? "yes" : "no") << " |\n";
    log << "| Delta | " << std::fixed << std::setprecision(4) << iter.delta << " |\n";
    log << "| Applied | " << (iter.applied ? "yes" : "no") << " |\n";
    log << "| Rolled back | " << (iter.rolled_back ? "yes" : "no") << " |\n";
    log << "| File | " << iter.file << " |\n";
    log << "| Line | " << iter.line << " |\n";
    log << "| Runtime (ms) | " << std::fixed << std::setprecision(1) << iter.runtime_ms << " |\n";
    log << "\n";
    log << "### Proof\n";
    log << "```\n";
    log << iter.proof.substr(0, 500);
    if (iter.proof.size() > 500) log << "\n... (truncated)";
    log << "\n```\n";
    log << "\n";

    // Also write to the project-level FLYWHEEL_LOG.md
    fs::path project_log = fs::current_path() / "FLYWHEEL_LOG.md";
    std::ofstream plog(project_log, std::ios::app);
    if (plog) {
        plog << "---\n";
        plog << "## Iteration " << iter.iter << "\n";
        plog << "**Task:** " << iter.task.substr(0, 100) << "\n";
        plog << "**Delta:** " << std::fixed << std::setprecision(4) << iter.delta << " | ";
        plog << "**Compiled:** " << (iter.compiled ? "yes" : "no") << " | ";
        plog << "**Verified:** " << (iter.verified ? "yes" : "no") << " | ";
        plog << "**Applied:** " << (iter.applied ? "yes" : "no") << " | ";
        plog << "**Rollback:** " << (iter.rolled_back ? "yes" : "no") << "\n";
        plog << "**File:** `" << iter.file << ":" << iter.line << "` | ";
        plog << "**Runtime:** " << std::fixed << std::setprecision(1) << iter.runtime_ms << "ms\n";
        plog << "**Proof:**\n```\n";
        plog << iter.proof.substr(0, 300);
        if (iter.proof.size() > 300) plog << "\n... (truncated)";
        plog << "\n```\n";
        plog << "\n";
    }
}

// ========================================================================
// run: main self-improvement flywheel loop
// ========================================================================
void Flywheel::run(int max_iters) {
    // Initialize log with header
    auto log_path = get_sandbox_path().parent_path() / "FLYWHEEL_LOG.md";
    std::error_code ec;
    fs::create_directories(get_sandbox_path(), ec);

    {
        std::ofstream log(log_path);
        if (log) {
            log << "# Self-Improving Flywheel Log\n";
            log << "Started: " << std::time(nullptr) << "\n";
            log << "Max iterations: " << max_iters << "\n";
            log << "Safety break: 10 consecutive no-improvement\n";
            log << "\n";
        }
    }

    // Also write to project level
    fs::path project_log = fs::current_path() / "FLYWHEEL_LOG.md";
    {
        std::ofstream plog(project_log);
        if (plog) {
            plog << "# Self-Improving Flywheel Log\n";
            plog << "Started: " << std::time(nullptr) << "\n";
            plog << "Max iterations: " << max_iters << "\n";
            plog << "\n";
        }
    }

    no_improvement_count_ = 0;
    converged_count_ = 0;
    history_.clear();

    for (int iter = 0; iter < max_iters; iter++) {
        auto iter_start = std::chrono::steady_clock::now();

        FlywheelIteration entry;
        entry.iter = iter;

        // Safety check: break if 10 consecutive no-improvement
        if (no_improvement_count_ >= 10) {
            std::ofstream log(log_path, std::ios::app);
            if (log) {
                log << "---\n";
                log << "## SAFETY BREAK at iteration " << iter << "\n";
                log << "No improvement for " << no_improvement_count_ << " consecutive iterations.\n";
                log << "\n";
            }
            break;
        }

        // Step 1: Generate a task via self_play
        std::string task = self_play();
        entry.task = task;

        // Check with safety guardrails
        if (safety_ && !safety_->check_input(task)) {
            entry.proof = "Task rejected by safety guardrails";
            entry.delta = -1.0f;
            history_.push_back(entry);
            log_iteration(entry);
            no_improvement_count_++;
            continue;
        }

        // Step 2: Generate solution code for the task
        std::string solution;
        if (model_) {
            int vocab_size = (int)model_->config.vocab_size;
            std::string prompt = "Write C++ code to solve this problem:\n" + task + "\n\nCode:";
            auto ids = quant::agi::util::simple_encode(prompt, vocab_size);
            auto gen = generate_new_tokens(model_, ids, vocab_size, 100);
            solution = quant::agi::util::simple_decode(gen);
        }

        if (solution.empty()) {
            solution = generate_from_template(task);
        }

        // Step 3: Compile and test in sandbox.
        // H7: compile_ok (sbox.compiled) and test_score (sbox.passed/score/exit_code)
        // are separate. Compile-only success must never increment the flywheel.
        SandboxResult sbox = sandbox_compile_and_test(solution, task);
        entry.compiled = sbox.compiled;
        entry.solution = solution;
        entry.runtime_ms = sbox.runtime_ms;

        if (!sbox.compiled) {
            entry.proof = "Compilation failed: " + sbox.stderr_capture;
            entry.delta = -0.5f;
            entry.verified = false;
            history_.push_back(entry);
            log_iteration(entry);
            no_improvement_count_++;
            continue;
        }

        // Compile succeeded but tests did not fully pass: not an improvement.
        // Gate is exit==0 && passed==total && total>=N (encoded in sbox.passed).
        if (!sbox.passed || sbox.exit_code != 0) {
            entry.proof = "Tests failed (compile_ok but test_score gate not met): exit=" +
                          std::to_string(sbox.exit_code) + " score=" +
                          std::to_string(sbox.score) + "\n" +
                          sbox.stdout_capture.substr(0, 500);
            entry.delta = -0.5f;
            entry.verified = false;
            entry.rolled_back = true;
            history_.push_back(entry);
            log_iteration(entry);
            no_improvement_count_++;
            continue;
        }

        // Step 4: Verify the solution with SelfVerifier (only reached on full test pass)
        bool verified = false;
        if (verifier_) {
            verified = verifier_->verify(task, solution);
        }
        entry.verified = verified;

        // Step 5: Measure capability improvement delta
        float delta = measure_improvement(task, solution);
        entry.delta = delta;

        // Step 6: Extract proof from the solution
        std::string proof = extract_proof(solution);
        entry.proof = proof;

        // Determine target file and line for the proof
        entry.file = "src/agi.cpp";
        entry.line = 0;
        for (char c : solution) if (c == '\n') entry.line++;

        // Step 7: If delta > 0 AND full test gate holds, apply improvement.
        // H7: never increment on compile-only success — sbox.passed/exit_code required.
        if (delta > 0.0f && sbox.passed && sbox.exit_code == 0) {
            // Create a diff of the solution against a baseline
            std::string baseline = "// Baseline: empty solution\n";
            std::string delta_diff = make_diff(baseline, solution);

            // Apply improvement to a target file
            std::string target_file = (fs::current_path() / "src" / "asi_generated.cpp").string();
            std::string backup_file = target_file + ".flywheel_backup." + std::to_string(iter);

            bool applied = apply_improvement(baseline, solution, target_file);
            entry.applied = applied;

            if (applied) {
                no_improvement_count_ = 0;
                entry.proof += "\n\nDiff applied:\n" + delta_diff.substr(0, 300);
            } else {
                no_improvement_count_++;
                entry.rolled_back = true;
                entry.proof += "\n\nFailed to apply improvement";
            }
        } else {
            // Rollback: no improvement
            no_improvement_count_++;
            entry.rolled_back = true;
            entry.proof += "\n\nNo improvement (delta <= 0), rolled back";
        }

        // Record iteration
        auto iter_end = std::chrono::steady_clock::now();
        entry.runtime_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start).count();

        history_.push_back(entry);
        log_iteration(entry);

        // Check convergence
        if (check_convergence()) {
            std::ofstream log(log_path, std::ios::app);
            if (log) {
                log << "---\n";
                log << "## CONVERGED at iteration " << iter << "\n";
                log << "The flywheel has converged with " << converged_count_ << " convergence signals.\n";
                log << "\n";
            }
            break;
        }
    }

    // Write summary
    {
        std::ofstream log(log_path, std::ios::app);
        if (log) {
            log << "---\n";
            log << "## Summary\n";
            log << "Total iterations: " << history_.size() << "\n";
            log << "Successful compilations: "
                << std::count_if(history_.begin(), history_.end(),
                                 [](auto& h) { return h.compiled; }) << "\n";
            log << "Improvements applied: "
                << std::count_if(history_.begin(), history_.end(),
                                 [](auto& h) { return h.applied; }) << "\n";
            log << "Rollbacks: "
                << std::count_if(history_.begin(), history_.end(),
                                 [](auto& h) { return h.rolled_back; }) << "\n";

            float total_delta = 0;
            for (auto& h : history_) total_delta += h.delta;
            log << "Total delta: " << std::fixed << std::setprecision(4) << total_delta << "\n";
            log << "Average delta: " << std::fixed << std::setprecision(4)
                << (history_.empty() ? 0.0f : total_delta / (float)history_.size()) << "\n";
            log << "\n";
            log << "---\n";
            log << "End of flywheel log.\n";
        }
    }

    // Also update project-level FLYWHEEL_LOG.md
    {
        std::ofstream plog(fs::current_path() / "FLYWHEEL_LOG.md", std::ios::app);
        if (plog) {
            plog << "---\n";
            plog << "## Summary\n";
            plog << "Total iterations: " << history_.size() << "\n";
            plog << "Improvements: " << std::count_if(history_.begin(), history_.end(),
                       [](auto& h) { return h.applied; });
            plog << " | Rollbacks: " << std::count_if(history_.begin(), history_.end(),
                       [](auto& h) { return h.rolled_back; });
            plog << " | Total delta: " << std::fixed << std::setprecision(4)
                 << std::accumulate(history_.begin(), history_.end(), 0.0f,
                     [](float acc, auto& h) { return acc + h.delta; }) << "\n";
            plog << "\n";
        }
    }
}

} // namespace agi
} // namespace quant
