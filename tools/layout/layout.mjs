// ForceAtlas2 layout for the union graph (all years). Usage: node layout.mjs in.json out.json [iterations] [settings-json]
// in.json: {"n": node count, "pairs": [a0, b0, a1, b1, ...]}; out.json: [x0, y0, x1, y1, ...]
import fs from 'node:fs';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';

const [, , inPath, outPath, itArg, setArg] = process.argv;
const iterations = Number(itArg || 600);
const { n, pairs } = JSON.parse(fs.readFileSync(inPath, 'utf8'));

// deterministic init: uniform disc (mulberry32)
let seed = 20260921;
const rnd = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const g = new Graph({ type: 'undirected' });
for (let i = 0; i < n; i++) {
  const r = Math.sqrt(rnd()) * 1000, a = rnd() * 2 * Math.PI;
  g.addNode(i, { x: r * Math.cos(a), y: r * Math.sin(a) });
}
for (let k = 0; k < pairs.length; k += 2) g.addEdge(pairs[k], pairs[k + 1]);

// 标准 FA2：不削弱枢纽的吸引力（outboundAttractionDistribution 会把叶子节点从所属企业旁推开），普通引力
const settings = {
  ...forceAtlas2.inferSettings(g),
  barnesHutOptimize: true,
  barnesHutTheta: 0.9,
  linLogMode: false,
  outboundAttractionDistribution: false,
  strongGravityMode: false,
  gravity: 1,
  scalingRatio: 10,
  ...(setArg ? JSON.parse(setArg) : {}),
};
console.log(`nodes ${g.order} edges ${g.size} iterations ${iterations}`, JSON.stringify(settings));
const t0 = Date.now(), chunk = 25;
for (let done = 0; done < iterations; done += chunk) {
  forceAtlas2.assign(g, { iterations: Math.min(chunk, iterations - done), settings });
  console.log(`  ${Math.min(done + chunk, iterations)}/${iterations}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
const out = new Array(2 * n);
g.forEachNode((k, a) => { out[2 * k] = +a.x.toFixed(2); out[2 * k + 1] = +a.y.toFixed(2); });
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(`wrote ${outPath} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
