import { wasm } from '@rollup/plugin-wasm';

export default [{
  input: 'index.js',
  output: {
    file: 'publish/dist/strprox.mjs',
    format: 'es',
    name: 'strprox'
  },
  plugins: [
    wasm({
        maxFileSize: 10e6,
    })
  ]
}];