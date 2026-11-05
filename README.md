# 論文（arXiv 2604.02334）參考實作計畫

 holos.py，純 Python + numpy，離線可跑。

## Installation 
`git clone https://github.com/ovidwu115/holos`
`pip install numpy`


## Usage 

跑完整流程，

`python holos.py demo`

跑經濟模擬。

`python holos.py sim`

跑 bench

`python bench_nuwa.py`

```
  seeds    50,000     0.3s
  seeds   100,000     0.6s
  seeds   149,999     0.9s
  seeds   199,997     1.3s
  seeds   249,995     1.7s
  seeds   299,992     2.1s
  seeds   349,991     2.6s
  seeds   399,983     3.0s
  seeds   449,978     3.4s
  seeds   499,973     3.8s
  seeds   549,964     4.2s
  seeds   599,958     4.7s
  seeds   649,952     5.1s
  seeds   699,937     5.6s
  seeds   749,934     6.1s
  seeds   799,927     6.5s
  seeds   849,916     7.0s
  seeds   899,907     7.5s
  seeds   949,897     8.0s
  seeds   999,880     8.5s
  seeds 1,000,000     8.6s

N=1,000,000  create 8.6s (116,819 seeds/s, no LLM)
disk: 193.4 MB  (193 B/agent; 10M est 1.93 GB)
RSS: 1338 MB total (vector index 1024 MB = 1024 B/agent)

population | point lookup mean/P99 (ms) | top-10 retrieval mean/P99 (ms)
        10 |   0.009 /   0.022        |     0.01 /     0.03
       100 |   0.009 /   0.018        |     0.01 /     0.02
     1,000 |   0.012 /   0.025        |     0.04 /     0.06
    10,000 |   0.011 /   0.018        |     0.46 /     0.82
   100,000 |   0.013 /   0.023        |     7.34 /     7.78
 1,000,000 |   0.014 /   0.021        |    74.96 /    84.18

JIT instantiate (profile fetch + S-MMR K=5, 40-tool pool): mean 0.31 ms, P99 0.38 ms
final RSS 1352 MB
```
