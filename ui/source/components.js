// Componentes presentacionales de la interfaz (en español).
// Usa htm para escribir JSX sin paso de compilación.

import React from 'react';
import {Box, Text} from 'ink';
import Spinner from 'ink-spinner';
import htm from 'htm';

const html = htm.bind(React.createElement);

export function Header() {
	return html`
		<${Box} flexDirection="column" marginBottom=${1}>
			<${Text} color="cyan" bold>
				LlamaVino${' '}
				<${Text} dimColor>· GGUF + OpenVINO en Intel Iris Xe<//>
			<//>
		<//>
	`;
}

// Barra de estado inferior: muestra SIEMPRE los valores de cada opción.
export function StatusBar({settings, status}) {
	const s = settings;
	const item = (label, value) =>
		html`<${Text}>${'  '}${label}:<${Text} color="cyan">${value}<//><//>`;
	return html`
		<${Box} flexDirection="column" marginTop=${1}>
			<${Box}>
				<${Text} backgroundColor="cyan" color="black"> LlamaVino <//>
				${item('modelo', s.modelName)}
				${item('disp', s.device)}
				${item('temp', s.temperature.toFixed(2))}
				${item('top_p', s.top_p.toFixed(2))}
				${item('top_k', String(s.top_k))}
				${item('pen_p', (s.presence_penalty ?? 0).toFixed(2))}
				${item('pen_f', (s.frequency_penalty ?? 0).toFixed(2))}
				${item('stop', (s.stop_strings && s.stop_strings.length) ? String(s.stop_strings.length) : '—')}
				${item('máx', String(s.max_new_tokens))}
				${item('stream', s.stream ? 'on' : 'off')}
			<//>
			${status
				? html`<${Text} dimColor>${status}<//>`
				: null}
		<//>
	`;
}

export function Loading({text}) {
	return html`
		<${Box}>
			<${Text} color="cyan"><${Spinner} type="dots" /><//>
			<${Text}> ${text}<//>
		<//>
	`;
}

// Lista de mensajes del chat con roles coloreados.
export function Messages({messages, streaming}) {
	return html`
		<${Box} flexDirection="column">
			${messages.map((m, i) =>
				html`<${Bubble} key=${i} role=${m.role} content=${m.content} />`,
			)}
			${streaming !== null
				? html`<${Bubble} role="assistant" content=${streaming} pending />`
				: null}
		<//>
	`;
}

function Bubble({role, content, pending}) {
	const labels = {user: 'Tú', assistant: 'LlamaVino', system: 'Sistema', tool: 'Herramienta'};
	const colors = {user: 'blue', assistant: 'magenta', system: 'yellow', tool: 'green'};
	return html`
		<${Box} flexDirection="column" marginBottom=${1}>
			<${Text} color=${colors[role] || 'white'} bold>${labels[role] || role}<//>
			<${Text}>${content}${pending ? html`<${Text} dimColor>▌<//>` : ''}<//>
		<//>
	`;
}
