#!/usr/bin/env node
// Punto de entrada de la interfaz Ink de LlamaVino.
// Lanza el motor Python (LlamaVino.py --serve) y monta la app de React.

import process from 'node:process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import React from 'react';
import {render} from 'ink';
import htm from 'htm';
import App from './app.js';

const html = htm.bind(React.createElement);

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..');
const script = path.join(repoRoot, 'LlamaVino.py');
// Permite override del intérprete (p.ej. una venv) sin tocar el código.
const python = process.env.LLAMAVINO_PYTHON || 'python';

render(html`<${App} python=${python} script=${script} cwd=${repoRoot} />`);
