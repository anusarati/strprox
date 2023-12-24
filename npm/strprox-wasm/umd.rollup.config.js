import { wasm } from '@rollup/plugin-wasm';

export default [{
  input: 'pkg/strprox.js',
  output: {
    file: 'publish/dist/umd/strprox.cjs',
    format: 'umd',
    name: 'strprox'
  },
  plugins: [
    wasm({
      maxFileSize: 10e6,
    })
  ]
},
];