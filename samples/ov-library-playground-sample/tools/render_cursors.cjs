// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Authoring only: node tools/render_cursors.cjs (sharp 0.35.5 on NODE_PATH).
// The application ships the small PNGs and needs no SVG renderer at runtime.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const folder = path.resolve(__dirname, '../data/cursors');

async function main() {
    const manifest = JSON.parse(fs.readFileSync(path.join(folder, 'manifest.json')));
    manifest.derived = {};
    for (const name of ['hand', 'hand-grab']) {
        const original = fs.readFileSync(path.join(folder, `${name}.svg`), 'utf8');
        const paths = original.match(/<path\s[^>]*\/>/g);
        if (!paths?.length) throw new Error(`No source paths in ${name}`);
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="0 0 28 28">
          <g transform="translate(2 2)" fill="none" stroke-linecap="round" stroke-linejoin="round">
            <g stroke="#111a22" stroke-opacity="0.82" stroke-width="3.2">${paths.join('')}</g>
            <g stroke="#e5eceb" stroke-width="1.5">${paths.join('')}</g>
          </g>
        </svg>`;
        const png = await sharp(Buffer.from(svg)).png().toBuffer();
        fs.writeFileSync(path.join(folder, `${name}.png`), png);
        manifest.derived[`${name}.png`] = {
            source: `${name}.svg`, sha256: crypto.createHash('sha256').update(png).digest('hex'),
            renderer: `sharp ${sharp.versions.sharp}`, size: 224,
            adaptation: 'Original paths; rounded 1.5-unit pale outline over a 3.2-unit dark contrast edge; 2-unit transparent padding.'
        };
    }
    fs.writeFileSync(path.join(folder, 'manifest.json'), JSON.stringify(manifest, null, 2)+'\n');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
