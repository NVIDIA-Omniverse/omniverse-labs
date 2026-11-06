// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Authoring only: rasterize the pinned SVG for OVUI's image button.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const folder = path.resolve(__dirname, '../data/icons');
async function main() {
    const svg = fs.readFileSync(path.join(folder, 'info.svg'), 'utf8');
    const pixels = await sharp(Buffer.from(svg.replace('currentColor', '#ffffff')), {density: 288})
        .resize(96, 96).png().toBuffer();
    fs.writeFileSync(path.join(folder, 'info.png'), pixels);
    const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
    fs.writeFileSync(path.join(folder, 'manifest.json'), JSON.stringify({
        source: 'https://github.com/lucide-icons/lucide',
        revision: 'd672fc43dbf60962d9d829e35bc3b3e5ed89a178',
        license: 'ISC and MIT (Feather-derived info icon)',
        source_sha256: sha(Buffer.from(svg)),
        license_sha256: sha(fs.readFileSync(path.join(folder, 'LICENSE.txt'))),
        derived: {file: 'info.png', sha256: sha(pixels), renderer: `sharp ${sharp.versions.sharp}`,
            size: 96, adaptation: 'Original 24-unit paths and 2-unit strokes; white stroke for UI tinting; rendered at 4x.'}
    }, null, 2)+'\n');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
