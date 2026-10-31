import { dirname } from 'path';
import { fileURLToPath } from 'url';
import { FlatCompat } from '@eslint/eslintrc';

const compat = new FlatCompat({ baseDirectory: dirname(fileURLToPath(import.meta.url)) });

// The landing site is linted with the Next.js rule set. `effgen-docs/` is a
// separate package with its own flat config, the build outputs are generated,
// and `build_plan/` is local working material that is not part of the site, so
// none of them is linted from here.
const config = [
  {
    ignores: [
      '.next/**',
      'out/**',
      'next-env.d.ts',
      'effgen-docs/**',
      'public/docs/**',
      'node_modules/**',
      'build_plan/**',
      'scripts/**',
    ],
  },
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
];

export default config;
