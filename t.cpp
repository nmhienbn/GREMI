#include <bits/stdc++.h>
using namespace std;

const int LIMIT = 1e3;
vector<int> primes;
bool is_prime[LIMIT + 1];

void sieve() {
    fill(is_prime, is_prime + LIMIT + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= LIMIT; ++i) {
        if (is_prime[i]) {
            primes.push_back(i);
            for (int j = i * 2; j <= LIMIT; j += i) {
                is_prime[j] = false;
            }
        }
    }
}

vector<tuple<long long, long long, long long>> generate_test_cases(int t, long long max_n = 1e11) {
    vector<tuple<long long, long long, long long>> test_cases;
    double total_sqrt_p = 0;
    random_device rd;
    mt19937_64 gen(rd());
    
    for (int i = 0; i < t; ++i) {
        int p = primes[gen() % primes.size()];
        if (p <= 2) continue;
        total_sqrt_p += sqrt(p);
        if (total_sqrt_p > 320000) break;
        uniform_int_distribution<long long> dist_p(2, p - 1);
        long long n = dist_p(gen);
        uniform_int_distribution<long long> dist_k(1, n - 1);
        long long k = dist_k(gen);
        test_cases.emplace_back(n, k, p);
    }
    return test_cases;
}

void write_test_file(const string &filename, int t) {
    sieve();
    auto test_cases = generate_test_cases(t);
    ofstream fout(filename);
    fout << test_cases.size() << "\n";
    int sum = 0;
    for (const auto &[n, k, p] : test_cases) {
        fout << n << " " << k << " " << p << "\n";
        sum += sqrt(p);
    }
    cout << sum << "\n";
    fout.close();
}

int main() {
    write_test_file("CKN4.INP", 100000);
    return 0;
}
