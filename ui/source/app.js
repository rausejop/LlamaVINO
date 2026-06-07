// Aplicación Ink principal de LlamaVino (toda la interfaz en español).
//
// Pantallas (máquina de estados):
//   connecting -> hub (ajustes con flechas) -> picker -> loading -> chat
//                   └── config (pantalla /configuration) ──┘
//
// El hub muestra todas las opciones (las mismas que la línea de comandos) y se
// navega con ↑/↓ para elegir y ←/→ para cambiar el valor. La pantalla
// «config» (abierta con «/configuration» en el chat o desde el hub) edita los
// parámetros comunes de un LLM con descripciones: temperatura, top-p, top-k,
// penalizaciones de presencia/frecuencia, máx. tokens y secuencias de parada.
// Dentro del chat los ajustes rápidos también se cambian con comandos «/».

import React, {useState, useEffect, useRef} from 'react';
import {Box, Text, useApp, useInput} from 'ink';
import TextInput from 'ink-text-input';
import htm from 'htm';
import {Backend} from './backend.js';
import {Header, StatusBar, Loading, Messages} from './components.js';

const html = htm.bind(React.createElement);

const HUB_FIELDS = [
	{key: 'chat', label: '▶  Iniciar chat', kind: 'action'},
	{key: 'model', label: 'Modelo', kind: 'model'},
	{key: 'device', label: 'Dispositivo', kind: 'choice'},
	{key: 'temperature', label: 'Temperatura', kind: 'num'},
	{key: 'top_p', label: 'top_p', kind: 'num'},
	{key: 'top_k', label: 'top_k', kind: 'num'},
	{key: 'max_new_tokens', label: 'Máx. tokens nuevos', kind: 'num'},
	{key: 'stream', label: 'Streaming', kind: 'bool'},
	{key: 'config', label: '⚙  Configuración avanzada…', kind: 'action'},
	{key: 'quit', label: 'Salir', kind: 'action'},
];

// Parámetros comunes de un LLM editados en la pantalla /configuration. Cada
// campo lleva su rango, paso y una breve descripción en español (misma fuente
// de verdad que GEN_PARAM_SPECS en LlamaVino.py).
const CONFIG_FIELDS = [
	{
		key: 'temperature', label: 'Temperatura', kind: 'num',
		step: 0.1, min: 0, max: 2, dp: 2,
		help: 'Aleatoriedad. 0–0.3 preciso · 0.7 equilibrado · 1.0+ creativo. 0 = greedy.',
	},
	{
		key: 'top_p', label: 'Top-P (nucleus)', kind: 'num',
		step: 0.05, min: 0, max: 1, dp: 2,
		help: 'Prob. acumulada de candidatas (0.9 típico). Mueve esto o la temperatura, no ambos.',
	},
	{
		key: 'top_k', label: 'Top-K', kind: 'num',
		step: 5, min: 0, max: 1000, dp: 0,
		help: 'Número fijo de palabras candidatas (p. ej. 40). 0 = sin límite.',
	},
	{
		key: 'presence_penalty', label: 'Penalización de presencia', kind: 'num',
		step: 0.1, min: -2, max: 2, dp: 2,
		help: 'Castiga palabras por haber aparecido (fomenta temas nuevos). Positivo bajo: 0.1–0.5.',
	},
	{
		key: 'frequency_penalty', label: 'Penalización de frecuencia', kind: 'num',
		step: 0.1, min: -2, max: 2, dp: 2,
		help: 'Castiga palabras según cuánto se repiten. Positivo bajo evita repetir frases.',
	},
	{
		key: 'max_new_tokens', label: 'Máximo de tokens', kind: 'num',
		step: 64, min: 16, max: 32768, dp: 0,
		help: 'Longitud máxima de la respuesta (1 token ≈ 4 caracteres).',
	},
	{
		key: 'stop_strings', label: 'Secuencias de parada', kind: 'text',
		help: "Cadenas que detienen la generación al aparecer (p. ej. 'Usuario:'). Separa con comas.",
	},
];

// Comandos «/» del chat. Reflejan el conjunto de Claude Code, adaptados al
// motor local (GGUF + OpenVINO). El primer nombre de cada uno es el canónico.
const CHAT_COMMANDS = [
	{names: ['/help', '/ayuda'], help: 'Muestra la lista de comandos.'},
	{names: ['/clear', '/limpiar'], help: 'Reinicia la conversación.'},
	{names: ['/compact', '/compactar'], help: 'Resume la conversación para ahorrar contexto.'},
	{names: ['/config', '/configuration', '/configuración', '/configuracion'], help: 'Editor de parámetros.'},
	{names: ['/model', '/models', '/modelo'], help: 'Cambia de modelo.'},
	{names: ['/save', '/guardar'], help: 'Guarda el último código: /save [ruta].'},
	{names: ['/cost', '/coste'], help: 'Uso de tokens de la sesión.'},
	{names: ['/status', '/estado'], help: 'Modelo, dispositivo y ajustes activos.'},
	{names: ['/gguf'], help: 'Cabecera del GGUF: /gguf [filtro] · /gguf tokens vuelca arrays.'},
	{names: ['/ir'], help: 'Información del modelo OpenVINO IR (ficheros, config, rt_info).'},
	{names: ['/doctor'], help: 'Diagnóstico de OpenVINO y dispositivos.'},
	{names: ['/mcp'], help: 'Lista los servidores MCP configurados.'},
	{names: ['/menu', '/ajustes'], help: 'Vuelve a la pantalla de ajustes.'},
	{names: ['/exit', '/quit', '/salir'], help: 'Sale de LlamaVino.'},
];

// Atajos rápidos (toman un argumento numérico); se muestran en el menú «/».
const QUICK_COMMANDS = [
	{names: ['/temp'], help: 'Atajo: fija la temperatura (/temp 0.7).'},
	{names: ['/top_p'], help: 'Atajo: fija top_p (/top_p 0.9).'},
	{names: ['/top_k'], help: 'Atajo: fija top_k (/top_k 40).'},
	{names: ['/max'], help: 'Atajo: fija máx. tokens (/max 512).'},
	{names: ['/stream'], help: 'Atajo: activa/desactiva streaming (/stream off).'},
];

const ALL_COMMANDS = [...CHAT_COMMANDS, ...QUICK_COMMANDS];

const resolveCommand = token => {
	const t = token.toLowerCase();
	const match = CHAT_COMMANDS.find(c => c.names.includes(t));
	return match ? match.names[0] : null;
};

// Comandos cuyo nombre canónico o algún alias empieza por lo tecleado.
const slashMatches = draft => {
	const t = draft.toLowerCase();
	return ALL_COMMANDS.filter(c => c.names.some(n => n.startsWith(t)));
};

// El menú «/» está activo mientras se teclea el nombre del comando (sin espacio).
const slashMenuOpen = draft =>
	draft.startsWith('/') && !draft.includes(' ') && slashMatches(draft).length > 0;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const round2 = v => Math.round(v * 100) / 100;

// Primer de sistema: enseña al modelo a marcar el nombre del archivo que quiere
// escribir, igual que FILE_PRIMER en LlamaVino.py.
const FILE_PRIMER =
	'Puedes crear archivos en el espacio de trabajo del usuario. Cuando generes ' +
	'el contenido completo de un archivo, escríbelo en un bloque de código cercado ' +
	'e indica el nombre del archivo en la línea de apertura, por ejemplo:\n' +
	"```python hola.py\nprint('hola')\n```\nUsa rutas relativas dentro del proyecto.";

const FILENAME_RE = /[\w./\\-]+\.[A-Za-z0-9]{1,8}/;

// Extrae bloques de código cercados y, para cada uno, el mejor nombre de archivo
// (de la línea de info, de un comentario inicial o del texto previo).
function extractCodeBlocks(text) {
	const blocks = [];
	const re = /```([^\n]*)\n([\s\S]*?)```/g;
	let m;
	while ((m = re.exec(text)) !== null) {
		const info = (m[1] || '').trim();
		const code = m[2] || '';
		const lang = info ? info.split(/\s+/)[0] : '';
		blocks.push({lang, code, filename: detectFilename(info, code, text.slice(0, m.index))});
	}
	return blocks;
}

function detectFilename(info, code, preceding) {
	const kv = info.match(/(?:path|file|filename|title|name)\s*=\s*["']?([^\s"']+)/i);
	if (kv) return kv[1];
	for (const tok of info.split(/[\s:]+/)) {
		if (tok && FILENAME_RE.test(tok) && /^[\w./\\-]+$/.test(tok)) {
			const full = tok.match(FILENAME_RE);
			if (full && full[0] === tok) return tok;
		}
	}
	const first = (code.split('\n')[0] || '')
		.replace(/^\s*(?:#|\/\/|--|<!--|\/\*|;)\s*/, '')
		.replace(/-->|\*\/$/g, '')
		.replace(/^(?:file|archivo|fichero)\s*:\s*/i, '')
		.trim();
	const firstMatch = first.match(FILENAME_RE);
	if (firstMatch && firstMatch[0] === first) return first;
	const tail = preceding.slice(-200);
	const cands = tail.match(new RegExp(FILENAME_RE.source, 'g'));
	return cands ? cands[cands.length - 1] : null;
}
const baseName = p => (p ? p.split(/[\\/]/).pop() : '(sin seleccionar)');

export default function App({python, script, cwd}) {
	const {exit} = useApp();
	const backendRef = useRef(null);
	const settingsRef = useRef(null);
	const loadedKeyRef = useRef(null);

	const [phase, setPhase] = useState('connecting');
	const [models, setModels] = useState([]);
	const [devices, setDevices] = useState(['AUTO', 'GPU', 'CPU']);
	const [settings, setSettings] = useState({
		modelPath: null,
		modelName: '(sin seleccionar)',
		device: 'AUTO',
		temperature: 0.7,
		top_p: 0.9,
		top_k: 40,
		presence_penalty: 0.0,
		frequency_penalty: 0.0,
		stop_strings: [],
		max_new_tokens: 512,
		stream: true,
	});
	const [hubIndex, setHubIndex] = useState(0);
	const [pickIndex, setPickIndex] = useState(0);
	const [configIndex, setConfigIndex] = useState(0);
	const [editingStops, setEditingStops] = useState(false);
	const [stopsDraft, setStopsDraft] = useState('');
	const configReturnRef = useRef('hub');
	const [messages, setMessages] = useState([]);
	const [draft, setDraft] = useState('');
	const [suggestIndex, setSuggestIndex] = useState(0); // resaltado del menú «/»
	const [streaming, setStreaming] = useState(null);
	const [status, setStatus] = useState('Conectando con el motor…');
	const generatingRef = useRef(false);
	const costRef = useRef({turns: 0, tokens: 0, ms: 0}); // uso de la sesión (/cost)
	const compactingRef = useRef(false); // /compact en curso
	const deviceRef = useRef('AUTO');
	const lastResponseRef = useRef(''); // última respuesta (para /save)

	settingsRef.current = settings;
	deviceRef.current = settings.device;

	useEffect(() => {
		const be = new Backend({python, script, cwd});
		backendRef.current = be;

		be.on('Ready', () => {
			be.send('ListDevices', {'@id': 'dev'});
			be.send('ListModels', {'@id': 'mdl', models_dir: 'models'});
			setStatus('Elige opciones con ↑/↓ ←/→ · Enter para activar');
			setPhase('hub');
		});
		be.on('Devices', m => setDevices(['AUTO', ...m.items]));
		be.on('Models', m => setModels(m.items));
		be.on('Downloaded', m => {
			setSettings(s => ({...s, modelPath: m.path, modelName: baseName(m.path)}));
			be.send('ListModels', {'@id': 'mdl', models_dir: 'models'});
			loadModel(m.path);
		});
		be.on('Loaded', m => {
			loadedKeyRef.current = `${settingsRef.current.modelPath}|${settingsRef.current.device}`;
			setSettings(s => ({...s, device: m.device}));
			setStatus(`Modelo listo en ${m.device} (${(m.ms / 1000).toFixed(1)} s)`);
			setPhase('chat');
		});
		be.on('Token', m => setStreaming(prev => (prev ?? '') + m.text));
		be.on('Done', m => {
			generatingRef.current = false;
			setStreaming(null);
			const ot = m.usage ? m.usage.output_tokens : 0;
			const c = costRef.current;
			c.turns += 1;
			c.tokens += ot;
			c.ms += m.ms || 0;
			if (compactingRef.current) {
				// /compact: sustituye el historial por un único resumen.
				compactingRef.current = false;
				setMessages([
					{role: 'system', content: 'Resumen de la conversación previa:\n' + m.text},
				]);
				setStatus('Conversación compactada.');
				return;
			}
			setMessages(prev => [...prev, {role: 'assistant', content: m.text}]);
			lastResponseRef.current = m.text;
			setStatus(`Respuesta en ${(m.ms / 1000).toFixed(1)} s · ${ot} tok`);
			// Autoguarda los bloques de código que traen nombre de archivo.
			const named = extractCodeBlocks(m.text).filter(b => b.filename);
			for (const b of named) {
				be.send('WriteFile', {'@id': 'wf', path: b.filename, content: b.code});
			}
		});
		be.on('FileWritten', m => {
			const name = (m.path || '').split(/[\\/]/).pop();
			setStatus(`✓ Archivo guardado: ${name} (${m.bytes} bytes)`);
		});
		be.on('GgufInfo', m => {
			const filt = m.filter
				? ` · filtro: ${m.filter} (${m.matched}/${m.total} claves)`
				: '';
			const header =
				`Cabecera GGUF · v${m.version} · ${m.tensor_count} tensores · ` +
				`${m.kv_count} claves · arquitectura: ${m.architecture}${filt}`;
			const rows = m.rows || [];
			const lines = rows.length
				? rows.map(r => {
						if (r.array) {
							// Vuelca el array (p. ej. /gguf tokens); puede venir acotado.
							const a = r.array;
							const nota = a.shown < a.len ? ` (mostrando ${a.shown} de ${a.len})` : '';
							return `• ${r.key} — ${a.len} elementos${nota}:\n    ${a.values.join(', ')}`;
						}
						return `• ${r.key} = ${r.value}` + (r.meaning ? `\n    ${r.meaning}` : '');
				  })
				: [`(sin coincidencias para «${m.filter}»)`];
			setMessages(prev => [...prev, {role: 'system', content: header + '\n' + lines.join('\n')}]);
			setStatus('Cabecera GGUF mostrada.');
		});
		be.on('IrInfo', m => {
			const gib = b => `${(b / 1024 ** 3).toFixed(2)} GiB`;
			const header = `Modelo OpenVINO IR · ${m.name} · ${gib(m.total_bytes)} · arquitectura: ${m.architecture}`;
			const ficheros = (m.files || []).map(f => `  ${f.name}  ${gib(f.bytes)}`);
			const config = (m.config_rows || []).map(
				r => `• ${r.key} = ${r.value}` + (r.meaning ? `\n    ${r.meaning}` : ''));
			const rt = (m.rt_info || []).map(r => `  ${r.key} = ${r.value}`);
			const partes = [header, 'Ficheros:', ...ficheros, 'Configuración:', ...config];
			if (rt.length) partes.push('rt_info:', ...rt);
			setMessages(prev => [...prev, {role: 'system', content: partes.join('\n')}]);
			setStatus('Información IR mostrada.');
		});
		be.on('McpServers', m => {
			const list = (m.items || []).join(', ');
			setStatus(list ? `Servidores MCP: ${list}` : 'No hay servidores MCP configurados.');
		});
		be.on('Error', m => {
			generatingRef.current = false;
			setStreaming(null);
			setStatus(`Error: ${m.message}`);
		});
		be.on('Fatal', m => setStatus(`Error fatal: ${m.message}`));

		return () => be.quit();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	function loadModel(pathOrName, engine = 'auto') {
		setStatus('Cargando modelo…');
		setPhase('loading');
		backendRef.current.send('Load', {
			model: pathOrName,
			device: settingsRef.current.device,
			engine, // 'auto' decide OpenVINO o llama.cpp (respaldo) al cargar
			n_gpu_layers: 'auto',
			'@id': 'load',
		});
	}

	function adjustField(field, dir) {
		setSettings(s => {
			const n = {...s};
			if (field === 'device') {
				const opts = devices.length ? devices : ['AUTO'];
				const i = Math.max(0, opts.indexOf(s.device));
				n.device = opts[(i + dir + opts.length) % opts.length];
			} else if (field === 'temperature') {
				n.temperature = Math.round(clamp(s.temperature + 0.1 * dir, 0, 2) * 100) / 100;
			} else if (field === 'top_p') {
				n.top_p = Math.round(clamp(s.top_p + 0.05 * dir, 0, 1) * 100) / 100;
			} else if (field === 'top_k') {
				n.top_k = clamp(s.top_k + 5 * dir, 0, 200);
			} else if (field === 'max_new_tokens') {
				n.max_new_tokens = clamp(s.max_new_tokens + 64 * dir, 16, 4096);
			} else if (field === 'stream') {
				n.stream = !s.stream;
			}
			return n;
		});
	}

	// Ajusta un parámetro numérico de la pantalla /configuration con ←/→.
	function adjustConfigField(dir) {
		const f = CONFIG_FIELDS[configIndex];
		if (f.kind !== 'num') return;
		setSettings(s => {
			const raw = clamp(s[f.key] + f.step * dir, f.min, f.max);
			return {...s, [f.key]: f.dp ? round2(raw) : raw};
		});
	}

	function openConfig(returnPhase) {
		configReturnRef.current = returnPhase;
		setConfigIndex(0);
		setEditingStops(false);
		setPhase('config');
	}

	function leaveConfig() {
		setEditingStops(false);
		setPhase(configReturnRef.current || 'hub');
	}

	function commitStops(value) {
		const list = value
			.split(',')
			.map(s => s.trim())
			.filter(Boolean);
		setSettings(s => ({...s, stop_strings: list}));
		setEditingStops(false);
		setStatus(`Secuencias de parada: ${list.length ? list.join(', ') : '(ninguna)'}`);
	}

	function startChat() {
		const s = settingsRef.current;
		if (!s.modelPath) {
			setPickIndex(0);
			setPhase('picker');
			return;
		}
		const key = `${s.modelPath}|${s.device}`;
		if (loadedKeyRef.current === key) setPhase('chat');
		else loadModel(s.modelPath);
	}

	function doExit() {
		backendRef.current?.quit();
		exit();
	}

	// --- entrada de teclado por pantalla ---------------------------------- //
	useInput(
		(input, key) => {
			if (phase === 'hub') {
				if (key.upArrow) setHubIndex(i => (i - 1 + HUB_FIELDS.length) % HUB_FIELDS.length);
				else if (key.downArrow) setHubIndex(i => (i + 1) % HUB_FIELDS.length);
				else if (key.leftArrow) adjustField(HUB_FIELDS[hubIndex].key, -1);
				else if (key.rightArrow) adjustField(HUB_FIELDS[hubIndex].key, 1);
				else if (key.return) {
					const f = HUB_FIELDS[hubIndex];
					if (f.key === 'chat') startChat();
					else if (f.key === 'model') {
						setPickIndex(0);
						setPhase('picker');
					} else if (f.key === 'config') openConfig('hub');
					else if (f.key === 'quit') doExit();
					else if (f.kind === 'bool' || f.kind === 'choice') adjustField(f.key, 1);
				} else if (input === 'q') doExit();
			} else if (phase === 'config') {
				if (editingStops) return; // el TextInput gestiona las teclas
				if (key.upArrow) setConfigIndex(i => (i - 1 + CONFIG_FIELDS.length) % CONFIG_FIELDS.length);
				else if (key.downArrow) setConfigIndex(i => (i + 1) % CONFIG_FIELDS.length);
				else if (key.leftArrow) adjustConfigField(-1);
				else if (key.rightArrow) adjustConfigField(1);
				else if (key.return) {
					if (CONFIG_FIELDS[configIndex].kind === 'text') {
						setStopsDraft((settingsRef.current.stop_strings || []).join(', '));
						setEditingStops(true);
					}
				} else if (key.escape || input === 'q') leaveConfig();
			} else if (phase === 'picker') {
				if (key.upArrow) setPickIndex(i => (i - 1 + models.length) % models.length);
				else if (key.downArrow) setPickIndex(i => (i + 1) % models.length);
				else if (key.escape || input === 'q') setPhase('hub');
				else if (key.return && models[pickIndex]) selectModel(models[pickIndex]);
			} else if (phase === 'chat') {
				if (slashMenuOpen(draft)) {
					// Menú de comandos «/» abierto: ↑/↓ selecciona, Tab completa,
					// Esc lo cierra. Enter lo gestiona submitChat (lo ejecuta).
					const matches = slashMatches(draft);
					if (key.upArrow) setSuggestIndex(i => (i - 1 + matches.length) % matches.length);
					else if (key.downArrow) setSuggestIndex(i => (i + 1) % matches.length);
					else if (key.tab) {
						const sel = matches[Math.min(suggestIndex, matches.length - 1)];
						setDraft(sel.names[0] + ' ');
						setSuggestIndex(0);
					} else if (key.escape) {
						setDraft('');
						setSuggestIndex(0);
					}
				} else if (key.escape) {
					setPhase('hub');
				}
			}
		},
		{isActive: phase !== 'connecting' && phase !== 'loading'},
	);

	function selectModel(row) {
		if (row.downloaded && row.path) {
			setSettings(s => ({...s, modelPath: row.path, modelName: baseName(row.path)}));
			settingsRef.current = {...settingsRef.current, modelPath: row.path};
			// Pista del registro: los marcados llamacpp van directos al respaldo.
			loadModel(row.path, row.engine === 'llamacpp' ? 'llamacpp' : 'auto');
		} else if (row.target) {
			setStatus(`Descargando ${row.name} (${row.size})…`);
			setPhase('loading');
			backendRef.current.send('Download', {target: row.target, models_dir: 'models', '@id': 'dl'});
		}
	}

	// Resetea el resaltado del menú cada vez que cambia lo tecleado.
	function onDraftChange(value) {
		setDraft(value);
		setSuggestIndex(0);
	}

	function submitChat(value) {
		// Con el menú «/» abierto, Enter ejecuta el comando resaltado.
		if (slashMenuOpen(value)) {
			const matches = slashMatches(value);
			const sel = matches[Math.min(suggestIndex, matches.length - 1)];
			setDraft('');
			setSuggestIndex(0);
			return handleSlash(sel.names[0]);
		}
		const v = value.trim();
		setDraft('');
		setSuggestIndex(0);
		if (!v) return;
		if (v.startsWith('/')) return handleSlash(v);
		if (generatingRef.current) return;
		const next = [...messages, {role: 'user', content: v}];
		setMessages(next);
		setStreaming('');
		generatingRef.current = true;
		const s = settingsRef.current;
		backendRef.current.send('Generate', {
			'@id': 'gen',
			history: [
				{role: 'system', content: FILE_PRIMER},
				...next.map(m => ({role: m.role, content: m.content})),
			],
			temperature: s.temperature,
			top_p: s.top_p,
			top_k: s.top_k,
			presence_penalty: s.presence_penalty,
			frequency_penalty: s.frequency_penalty,
			stop_strings: s.stop_strings,
			max_new_tokens: s.max_new_tokens,
		});
	}

	function handleSlash(v) {
		const [cmd, arg] = v.split(/\s+/, 2);
		const c = cmd.toLowerCase();
		switch (resolveCommand(cmd)) {
			case '/exit':
				return doExit();
			case '/menu':
				return setPhase('hub');
			case '/config':
				return openConfig('chat');
			case '/model':
				setPickIndex(0);
				return setPhase('picker');
			case '/save':
				return saveLastResponse(arg);
			case '/clear':
				setMessages([]);
				costRef.current = {turns: 0, tokens: 0, ms: 0};
				return setStatus('Conversación reiniciada.');
			case '/compact':
				return compactConversation();
			case '/cost':
				return showCost();
			case '/status':
				return showStatus();
			case '/gguf': {
				// Parsea «/gguf [filtro] [N]» (N = tope opcional de elementos).
				const piezas = v.slice(cmd.length).trim().split(/\s+/).filter(Boolean);
				let limit = null;
				if (piezas.length && /^\d+$/.test(piezas[piezas.length - 1])) {
					limit = parseInt(piezas.pop(), 10);
				}
				backendRef.current.send('GgufHeader', {
					'@id': 'gguf',
					filter: piezas.join(' '),
					limit,
				});
				return setStatus('Leyendo cabecera GGUF…');
			}
			case '/ir':
				backendRef.current.send('IrHeader', {'@id': 'ir'});
				return setStatus('Leyendo modelo IR…');
			case '/doctor':
				return showDoctor();
			case '/mcp':
				backendRef.current.send('McpListServers', {'@id': 'mcp'});
				return setStatus('Consultando servidores MCP…');
			case '/help':
				return setStatus(
					CHAT_COMMANDS.map(x => x.names[0]).join(' ') +
						' · atajos: /temp /top_p /top_k /max /stream',
				);
			default:
				break;
		}
		// Atajos rápidos (no forman parte de Claude Code, pero son cómodos).
		const num = Number.parseFloat(arg);
		if (c === '/temp' && !Number.isNaN(num)) adjustTo('temperature', clamp(num, 0, 2));
		else if (c === '/top_p' && !Number.isNaN(num)) adjustTo('top_p', clamp(num, 0, 1));
		else if (c === '/top_k' && !Number.isNaN(num)) adjustTo('top_k', clamp(num, 0, 200));
		else if (c === '/max' && !Number.isNaN(num)) adjustTo('max_new_tokens', clamp(num, 16, 4096));
		else if (c === '/stream') adjustTo('stream', arg !== 'off');
		else setStatus(`Comando no reconocido: ${cmd} — escribe /help`);
	}

	function adjustTo(field, value) {
		setSettings(s => ({...s, [field]: value}));
		setStatus(`${field} = ${value}`);
	}

	// /compact: pide al modelo que resuma el historial; el handler 'Done'
	// sustituye los mensajes por el resumen (compactingRef).
	function compactConversation() {
		if (generatingRef.current) return;
		if (!messages.length) return setStatus('No hay conversación que compactar.');
		const transcript = messages.map(m => `${m.role}: ${m.content}`).join('\n');
		compactingRef.current = true;
		generatingRef.current = true;
		setStreaming('');
		setStatus('Compactando conversación…');
		backendRef.current.send('Generate', {
			'@id': 'compact',
			history: [
				{
					role: 'user',
					content:
						'Resume en español, de forma concisa, la siguiente conversación, ' +
						'conservando hechos, decisiones y contexto importantes:\n\n' +
						transcript,
				},
			],
			temperature: 0,
			max_new_tokens: 512,
		});
	}

	// /save [ruta]: guarda código de la última respuesta. Con ruta explícita usa
	// el primer bloque (o el texto entero); si no, los bloques con nombre.
	function saveLastResponse(arg) {
		const text = lastResponseRef.current;
		if (!text) return setStatus('Aún no hay ninguna respuesta que guardar.');
		const blocks = extractCodeBlocks(text);
		const be = backendRef.current;
		if (arg && arg.trim()) {
			const code = blocks.length ? blocks[0].code : text;
			be.send('WriteFile', {'@id': 'wf', path: arg.trim(), content: code});
			return;
		}
		const named = blocks.filter(b => b.filename);
		if (!named.length) {
			return setStatus(
				blocks.length
					? 'Hay código pero sin nombre de archivo. Usa /save <ruta>.'
					: 'La última respuesta no contiene código.',
			);
		}
		for (const b of named) {
			be.send('WriteFile', {'@id': 'wf', path: b.filename, content: b.code});
		}
	}

	function showCost() {
		const c = costRef.current;
		const secs = c.ms / 1000;
		const tps = secs > 0 ? c.tokens / secs : 0;
		setStatus(
			`Sesión: ${c.turns} turnos · ${c.tokens} tok · ${secs.toFixed(1)} s · ` +
				`${tps.toFixed(1)} tok/s · modelo local (sin coste)`,
		);
	}

	function showStatus() {
		const s = settingsRef.current;
		setStatus(
			`Modelo ${s.modelName} · ${s.device} · openvino · temp ${s.temperature} · ` +
				`top_p ${s.top_p} · top_k ${s.top_k} · ` +
				`pen ${s.presence_penalty}/${s.frequency_penalty} · máx ${s.max_new_tokens}`,
		);
	}

	function showDoctor() {
		const gpu = devices.some(d => d.startsWith('GPU'));
		backendRef.current.send('ListDevices', {'@id': 'dev'});
		setStatus(
			`OpenVINO · dispositivos: ${devices.join(', ') || '(ninguno)'} · ` +
				`GPU: ${gpu ? 'sí' : 'no'}`,
		);
	}

	// --- render ----------------------------------------------------------- //
	let body;
	if (phase === 'connecting' || phase === 'loading') {
		body = html`<${Loading} text=${status} />`;
	} else if (phase === 'hub') {
		body = html`<${Hub} settings=${settings} index=${hubIndex} />`;
	} else if (phase === 'picker') {
		body = html`<${Picker} models=${models} index=${pickIndex} />`;
	} else if (phase === 'config') {
		body = html`
			<${Configuration}
				settings=${settings}
				index=${configIndex}
				editingStops=${editingStops}
				stopsDraft=${stopsDraft}
				onStopsChange=${setStopsDraft}
				onStopsSubmit=${commitStops}
			/>
		`;
	} else {
		const menuOpen = slashMenuOpen(draft);
		body = html`
			<${Box} flexDirection="column">
				<${Messages} messages=${messages} streaming=${streaming} />
				${menuOpen
					? html`<${SlashMenu} matches=${slashMatches(draft)} index=${suggestIndex} />`
					: null}
				<${Box}>
					<${Text} color="blue" bold>Tú › <//>
					<${TextInput}
						value=${draft}
						onChange=${onDraftChange}
						onSubmit=${submitChat}
						placeholder="escribe un mensaje o / para ver comandos"
					/>
				<//>
			<//>
		`;
	}

	return html`
		<${Box} flexDirection="column">
			<${Header} />
			${body}
			<${StatusBar} settings=${settings} status=${status} />
		<//>
	`;
}

function Hub({settings, index}) {
	const value = key => {
		switch (key) {
			case 'device':
				return settings.device;
			case 'temperature':
				return settings.temperature.toFixed(2);
			case 'top_p':
				return settings.top_p.toFixed(2);
			case 'top_k':
				return String(settings.top_k);
			case 'max_new_tokens':
				return String(settings.max_new_tokens);
			case 'stream':
				return settings.stream ? 'on' : 'off';
			case 'model':
				return settings.modelName;
			default:
				return '';
		}
	};
	return html`
		<${Box} flexDirection="column" borderStyle="round" borderColor="cyan" paddingX=${1}>
			<${Text} bold>Ajustes<//>
			${HUB_FIELDS.map((f, i) => {
				const active = i === index;
				const editable = f.kind === 'num' || f.kind === 'choice' || f.kind === 'bool';
				const val = value(f.key);
				return html`
					<${Box} key=${f.key}>
						<${Text} color=${active ? 'green' : undefined}>${active ? '› ' : '  '}<//>
						<${Text} color=${active ? 'cyan' : undefined} bold=${active}>${f.label.padEnd(20)}<//>
						<${Text} color=${active && editable ? 'yellow' : 'gray'}>
							${val ? (active && editable ? `‹ ${val} ›` : val) : ''}
						<//>
					<//>
				`;
			})}
			<${Text} dimColor>↑/↓ moverse · ←/→ cambiar · Enter activar · q salir<//>
		<//>
	`;
}

function Picker({models, index}) {
	return html`
		<${Box} flexDirection="column" borderStyle="round" borderColor="cyan" paddingX=${1}>
			<${Text} bold>Modelos disponibles<//>
			${models.length === 0
				? html`<${Text} dimColor>cargando…<//>`
				: models.map((m, i) => {
						const active = i === index;
						const mark = m.downloaded ? '✓' : '⬇';
						const motor =
							m.engine === 'llamacpp'
								? {txt: 'llama.cpp', col: 'yellow'}
								: m.engine === 'openvino'
									? {txt: 'OpenVINO', col: 'green'}
									: {txt: 'auto', col: 'gray'};
						return html`
							<${Box} key=${m.name}>
								<${Text} color=${active ? 'green' : undefined}>${active ? '› ' : '  '}<//>
								<${Text} color=${m.downloaded ? 'green' : 'yellow'}>${mark} <//><${Text} color="green" bold=${true}>${(m.recomendacion ? '#' + m.recomendacion : '  ').padEnd(4)}<//>
								<${Text} color=${m.highlight ? 'magenta' : active ? 'cyan' : undefined} bold=${active || m.highlight}>${((m.highlight ? '✦ ' : '') + m.name).padEnd(22)}<//><${Text} color=${({bartowski: 'cyan', OpenVINO: 'green', unsloth: 'magenta', TheBloke: 'yellow', MaziyarPanahi: 'blue', local: 'gray'}[m.publisher] || 'white')}>${(m.publisher || '').padEnd(13)}<//>
								<${Text} dimColor>${m.size.padStart(8)} <//>
								<${Text} color="magenta">${(m.format || 'GGUF').padEnd(5)}<//>
								<${Text} color=${motor.col}>${motor.txt.padEnd(10)}<//>
								<${Text} dimColor>${m.note}<//>
							<//>
						`;
				  })}
			<${Text} dimColor>↑/↓ · Enter seleccionar/descargar · q volver · ✦ recomendado unsloth · verde OpenVINO · amarillo llama.cpp<//>
		<//>
	`;
}

// Pantalla /configuration: edita todos los parámetros comunes de un LLM.
// Numéricos con ←/→; las secuencias de parada con Enter (entrada de texto).
function Configuration({
	settings,
	index,
	editingStops,
	stopsDraft,
	onStopsChange,
	onStopsSubmit,
}) {
	const fmt = (f, value) => {
		if (f.kind === 'text') {
			return value && value.length ? value.join(', ') : '(ninguna)';
		}
		return f.dp ? value.toFixed(f.dp) : String(value);
	};
	const active = CONFIG_FIELDS[index];
	return html`
		<${Box} flexDirection="column" borderStyle="round" borderColor="cyan" paddingX=${1}>
			<${Text} bold>Configuración de generación<//>
			${CONFIG_FIELDS.map((f, i) => {
				const sel = i === index;
				const val = fmt(f, settings[f.key]);
				const editing = sel && f.kind === 'text' && editingStops;
				return html`
					<${Box} key=${f.key}>
						<${Text} color=${sel ? 'green' : undefined}>${sel ? '› ' : '  '}<//>
						<${Text} color=${sel ? 'cyan' : undefined} bold=${sel}>${f.label.padEnd(26)}<//>
						${editing
							? html`<${TextInput} value=${stopsDraft} onChange=${onStopsChange} onSubmit=${onStopsSubmit} />`
							: html`<${Text} color=${sel ? 'yellow' : 'gray'}>${
									sel && f.kind === 'num' ? `‹ ${val} ›` : val
							  }<//>`}
					<//>
				`;
			})}
			<${Box} marginTop=${1}>
				<${Text} dimColor>${active.help}<//>
			<//>
			<${Text} dimColor>
				${editingStops
					? 'Escribe las secuencias separadas por comas · Enter confirmar'
					: '↑/↓ moverse · ←/→ ajustar · Enter editar texto · q volver'}
			<//>
		<//>
	`;
}

// Menú de autocompletado de comandos «/» (estilo Claude Code). Aparece sobre el
// input al teclear «/» y se filtra por prefijo conforme se escribe.
function SlashMenu({matches, index}) {
	const sel = Math.min(index, matches.length - 1);
	return html`
		<${Box} flexDirection="column" borderStyle="round" borderColor="cyan" paddingX=${1}>
			${matches.map((c, i) => {
				const active = i === sel;
				return html`
					<${Box} key=${c.names[0]}>
						<${Text} color=${active ? 'green' : undefined}>${active ? '› ' : '  '}<//>
						<${Text} color=${active ? 'cyan' : 'white'} bold=${active}>${c.names[0].padEnd(14)}<//>
						<${Text} dimColor>${c.help}<//>
					<//>
				`;
			})}
			<${Text} dimColor>↑/↓ seleccionar · Tab completar · Enter ejecutar · Esc cerrar<//>
		<//>
	`;
}
