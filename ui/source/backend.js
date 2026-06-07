// Puente con el motor Python (LlamaVino.py --serve).
//
// Habla el protocolo JSON-LD por líneas: cada mensaje es un documento con
// @context/@type/@id. El proceso Python se lanza con argumentos en array (sin
// shell) para evitar inyección de comandos.

import {spawn} from 'node:child_process';
import {EventEmitter} from 'node:events';
import readline from 'node:readline';

const JSONLD_CONTEXT = 'https://llamavino.dev/ns/v1';

export class Backend extends EventEmitter {
	constructor({python, script, cwd}) {
		super();
		// spawn con array de argumentos => sin interpretación de shell (seguro).
		this.proc = spawn(python, [script, '--serve'], {
			cwd,
			stdio: ['pipe', 'pipe', 'pipe'],
		});

		const rl = readline.createInterface({input: this.proc.stdout});
		rl.on('line', line => {
			const text = line.trim();
			if (!text) return;
			let msg;
			try {
				msg = JSON.parse(text);
			} catch {
				return; // ignora líneas que no sean JSON-LD válido
			}
			this.emit('message', msg);
			if (msg['@type']) this.emit(msg['@type'], msg);
		});

		this.proc.stderr.on('data', chunk => this.emit('log', chunk.toString()));
		this.proc.on('exit', code => this.emit('close', code));
		this.proc.on('error', err => this.emit('fail', err));
	}

	// Envía un documento JSON-LD de petición.
	send(type, fields = {}) {
		const doc = {'@context': JSONLD_CONTEXT, '@type': type, ...fields};
		this.proc.stdin.write(JSON.stringify(doc) + '\n');
	}

	quit() {
		try {
			this.send('Quit');
			this.proc.stdin.end();
		} catch {
			/* el proceso ya pudo haber terminado */
		}
	}
}
