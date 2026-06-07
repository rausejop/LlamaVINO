// Prueba del puente Node <-> Python sin TTY (verifica el protocolo JSON-LD).
// Carga el modelo 1B y genera una respuesta corta por streaming.
//
// Uso:  node test-backend.mjs

import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Backend} from './source/backend.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');
const script = path.join(repoRoot, 'LlamaVino.py');

const be = new Backend({python: 'python', script, cwd: repoRoot});
const MODEL = 'models/Llama-3.2-1B-Instruct-Q4_K_M.gguf';
let out = '';

be.on('Ready', () => {
	console.log('[Ready] motor iniciado');
	be.send('Load', {model: MODEL, device: 'AUTO', '@id': 'l1'});
});
be.on('Loaded', m => {
	console.log(`[Loaded] ${m.device} en ${(m.ms / 1000).toFixed(1)} s`);
	be.send('Generate', {
		'@id': 'g1',
		history: [{role: 'user', content: 'Di hola en 3 palabras'}],
		max_new_tokens: 20,
		temperature: 0,
	});
});
be.on('Token', m => {
	out += m.text;
});
be.on('Done', m => {
	console.log(`[Done] "${out}" en ${(m.ms / 1000).toFixed(1)} s · ${m.usage.output_tokens} tokens`);
	be.quit();
	process.exit(0);
});
be.on('Error', m => {
	console.error('[Error]', m.message);
	be.quit();
	process.exit(1);
});

setTimeout(() => {
	console.error('Timeout');
	be.quit();
	process.exit(2);
}, 120000);
