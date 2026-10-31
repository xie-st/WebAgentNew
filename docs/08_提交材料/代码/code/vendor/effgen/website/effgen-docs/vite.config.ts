import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Where the built docs are served from. Under a project page that is
  // /<repo>/docs/; the default keeps a root-hosted site working.
  base: process.env.DOCS_BASE_PATH || '/docs/',
  resolve: {
    alias: {
      // Both sites read their counts from one file. The landing site reaches it
      // through its own "@/" alias; this points at the same copy on disk, so a
      // number can never disagree between the two halves of the site.
      '@data': path.resolve(__dirname, '../data'),
      // One syntax highlighter for the whole site. The landing site reaches the
      // same file through its own "@/" alias, so a code sample is coloured by
      // the same tokenizer on both halves of the site.
      '@shared': path.resolve(__dirname, '../shared'),
    },
  },
  build: {
    outDir: path.resolve(__dirname, '../public/docs'),
    emptyOutDir: true,
  },
})
