import re,sys
n=1396; g=open('/tmp/zh_gold.txt').read().split()
st={}
for l in open('/tmp/tf1.txt'):
    m=re.match(r'\s*(\d+)\s+(-?\d+)\s+[\d.]+\s+[\d.]+\s*$',l)
    if m: st[int(m.group(1))]=m.group(2)
hit=[st.get(n-1+i)==g[i] for i in range(len(g))]
print("teacher-forced top-1 %d/%d"%(sum(hit),len(g)))
print("misses (i, engine, HF):", [(i,st.get(n-1+i),g[i]) for i,h in enumerate(hit) if not h])
