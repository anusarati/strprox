import { wasm } from '@rollup/plugin-wasm';

export default [{
  input: 'index.js',
  output: {
    file: 'publish/dist/strprox.js',
    format: 'umd',
    name: 'strprox'
  },
  plugins: [
    wasm({
        maxFileSize: 10e6,
    })
  ]
}];