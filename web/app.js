const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/cti');

const accountsEl = document.getElementById('accounts');
const simulateBtn = document.getElementById('simulate');
const focusSelectedBtn = document.getElementById('focus-selected');
const generateSelectedBtn = document.getElementById('generate-selected');
const selectedAccountChip = document.getElementById('selected-account-chip');
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const modelStatus = document.getElementById('model-status');
const backendStatus = document.getElementById('backend-status');
const accountCountEl = document.getElementById('account-count');
const highRiskCountEl = document.getElementById('high-risk-count');
const communityCountEl = document.getElementById('community-count');
const linkCountEl = document.getElementById('link-count');
const accountSummaryEl = document.getElementById('account-summary');
const relationshipSummaryEl = document.getElementById('relationship-summary');
const plainEnglishEl = document.getElementById('plain-english');
const graphShell = document.getElementById('graph-shell');
const graphTooltip = document.getElementById('graph-tooltip');
const graphSvg = d3.select('#graph-svg');
const waterfallSvg = d3.select('#waterfall-svg');

const fallbackFeatures = ['F3924', 'F3898', 'F1921', 'F1166', 'F2582', 'F2388', 'F1057', 'F3914', 'F2390', 'F270', 'F3912', 'F2137'];
const featureDescriptions = {
  F3924: 'Confirmed mule indicator',
  F321: 'Behavioral anchor F321',
  F3836: 'Velocity signal F3836',
  F2082: 'Behavioral anchor F2082',
};

const state = {
  featureNames: [],
  accounts: [],
  links: [],
  selectedAccountId: null,
  selectedLinkId: null,
  hoveredNodeId: null,
  hoveredLinkId: null,
  shapCache: new Map(),
  ringCache: new Map(),
  currentRingSummary: null,
  simulation: null,
  backendHealthy: null,
  pendingSubscription: null,
};

function hashString(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(31, hash) + value.charCodeAt(index) | 0;
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  return function next() {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function formatDecimal(value) {
  return Number.isFinite(value) ? value.toFixed(1) : '0.0';
}

function riskBandForCti(cti) {
  if (cti >= 80) return 'Critical';
  if (cti >= 55) return 'High';
  if (cti >= 30) return 'Medium';
  return 'Low';
}

function accountRiskColor(cti) {
  if (cti >= 80) return '#ff6b7c';
  if (cti >= 55) return '#ffbf5d';
  if (cti >= 30) return '#74a9ff';
  return '#36d6c3';
}

function getSelectedAccount() {
  return state.accounts.find((account) => account.account_id === state.selectedAccountId) || null;
}

function getSelectedLink() {
  return state.links.find((link) => link.id === state.selectedLinkId) || null;
}

function getRingMembers(account) {
  return state.accounts.filter((candidate) => candidate.community === account.community);
}

function createTimeseries(rng, cti, community) {
  const baseline = 1 + community * 0.2 + cti / 120;
  return Array.from({ length: 6 }, (_, index) => {
    const step = index / 5;
    return [
      Number((baseline + (rng() - 0.5) * 0.3 + step * 0.35).toFixed(3)),
      Number((baseline * 1.2 + (rng() - 0.5) * 0.4 + step * 0.5).toFixed(3)),
      Number((baseline * 0.9 + (rng() - 0.5) * 0.25 + (1 - step) * 0.2).toFixed(3)),
    ];
  });
}

function createAccount(accountId, rank) {
  const seed = hashString(accountId);
  const rng = mulberry32(seed);
  const cti = Number((22 + rng() * 68 + (rank % 5) * 1.6).toFixed(1));
  const community = 1 + (seed % 4);
  const featureValues = {};

  state.featureNames.forEach((featureName, index) => {
    const featureRng = mulberry32(seed + (index + 1) * 9973);
    let value = 0.1 + featureRng() * 0.7 + cti / 450;
    if (featureName === 'F3924') {
      value = cti > 68 ? 1 : 0;
    } else if (featureName === 'F321') {
      value = Number((0.2 + featureRng() * 1.5 + community * 0.12).toFixed(3));
    } else if (featureName === 'F3836') {
      value = Number((0.3 + featureRng() * 2.2 + cti / 80).toFixed(3));
    } else if (featureName === 'F2082') {
      value = Number((0.15 + featureRng() * 1.1 + rank * 0.03).toFixed(3));
    } else {
      value = Number(value.toFixed(3));
    }
    featureValues[featureName] = value;
  });

  return {
    account_id: accountId,
    cti,
    community,
    riskBand: riskBandForCti(cti),
    volume: Math.round(120 + rng() * 720),
    velocity: Number((0.3 + rng() * 3.6).toFixed(2)),
    dwellTime: Math.round(2 + rng() * 24),
    alerts: Math.round(rng() * 7),
    featureValues,
    timeseries: createTimeseries(rng, cti, community),
  };
}

function loadFeatureNames() {
  return fetch('/static/ensemble/selected_features.json')
    .then((response) => {
      if (!response.ok) {
        throw new Error(`feature list unavailable (${response.status})`);
      }
      return response.json();
    })
    .then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        state.featureNames = data;
        return;
      }
      state.featureNames = fallbackFeatures.slice();
    })
    .catch(() => {
      state.featureNames = fallbackFeatures.slice();
    });
}

function rebuildLinks() {
  const links = [];
  const byCommunity = new Map();

  state.accounts.forEach((account) => {
    const bucket = byCommunity.get(account.community) || [];
    bucket.push(account);
    byCommunity.set(account.community, bucket);
  });

  byCommunity.forEach((members, communityId) => {
    const ordered = [...members].sort((left, right) => right.cti - left.cti);
    ordered.forEach((source, index) => {
      const target = ordered[index + 1];
      if (!target) {
        return;
      }
      const gap = Math.abs(source.cti - target.cti);
      const similarity = Number(Math.max(0.18, 1 - gap / 100).toFixed(3));
      const syncScore = Number((0.35 + (communityId % 3) * 0.12 + (ordered.length > 2 ? 0.08 : 0)).toFixed(3));
      const corr = Number(Math.max(0, 0.9 - gap / 140).toFixed(3));
      links.push({
        id: `community-${communityId}-${source.account_id}-${target.account_id}`,
        source: source.account_id,
        target: target.account_id,
        community: communityId,
        kind: 'community',
        weight: Number((0.58 * similarity + 0.27 * syncScore + 0.15 * corr).toFixed(3)),
        similarity,
        sync_score: syncScore,
        corr,
      });
    });
  });

  const strongest = [...state.accounts].sort((left, right) => right.cti - left.cti).slice(0, Math.min(4, state.accounts.length));
  strongest.forEach((source, index) => {
    const target = state.accounts[(index + 1) % state.accounts.length];
    if (!source || !target || source.account_id === target.account_id) {
      return;
    }
    const gap = Math.abs(source.cti - target.cti);
    links.push({
      id: `bridge-${source.account_id}-${target.account_id}`,
      source: source.account_id,
      target: target.account_id,
      community: `${source.community}-${target.community}`,
      kind: 'bridge',
      weight: Number((0.3 + (100 - gap) / 300).toFixed(3)),
      similarity: Number(Math.max(0.08, 1 - gap / 120).toFixed(3)),
      sync_score: Number((0.22 + ((source.community + target.community) % 4) * 0.05).toFixed(3)),
      corr: Number(Math.max(0, 0.55 - gap / 170).toFixed(3)),
    });
  });

  state.links = links;
}

function addAccount(accountId, silent = false) {
  if (state.accounts.some((account) => account.account_id === accountId)) {
    return state.accounts.find((account) => account.account_id === accountId);
  }
  const account = createAccount(accountId, state.accounts.length);
  state.accounts.push(account);
  rebuildLinks();
  if (!silent) {
    renderAll();
  }
  return account;
}

function subscribe(accountId) {
  state.pendingSubscription = accountId;
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'subscribe', account_id: accountId }));
  }
}

function generateBrief(accountId) {
  const account = state.accounts.find((candidate) => candidate.account_id === accountId);
  return fetch('/briefs/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      account_id: accountId,
      include_shap: true,
      include_dtw: true,
      include_ring: true,
      notes: account ? `Focused on ${account.account_id} from community ${account.community}` : 'Demo brief',
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error('Failed to generate brief');
      }
      return response.blob();
    })
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `brief_${accountId}.pdf`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    })
    .catch(() => {
      alert('Failed to generate brief');
    });
}

function renderAccounts() {
  accountsEl.innerHTML = '';
  state.accounts
    .slice()
    .sort((left, right) => right.cti - left.cti)
    .forEach((account) => {
      const card = document.createElement('div');
      card.className = `account${account.account_id === state.selectedAccountId ? ' selected' : ''}`;
      card.onclick = () => selectAccount(account.account_id);

      card.innerHTML = `
        <div class="account-top">
          <div>
            <div class="account-id">${account.account_id}</div>
            <div class="account-meta">
              <span class="link-chip">Community ${account.community}</span>
              <span class="link-chip">${account.riskBand}</span>
            </div>
          </div>
          <span class="pill" style="padding:6px 10px; background:${accountRiskColor(account.cti)}22; border-color:${accountRiskColor(account.cti)}44;">CTI ${formatDecimal(account.cti)}</span>
        </div>
        <div class="risk-bar"><span style="width:${Math.max(8, Math.min(account.cti, 100))}%; background:${accountRiskColor(account.cti)};"></span></div>
        <div class="account-actions">
          <button type="button" class="secondary" data-focus="${account.account_id}">Focus</button>
          <button type="button" data-brief="${account.account_id}">Generate brief</button>
        </div>
      `;

      card.querySelector('[data-focus]')?.addEventListener('click', (event) => {
        event.stopPropagation();
        selectAccount(account.account_id);
      });
      card.querySelector('[data-brief]')?.addEventListener('click', (event) => {
        event.stopPropagation();
        generateBrief(account.account_id);
      });

      accountsEl.appendChild(card);
    });
}

function updateSummaryCards() {
  accountCountEl.textContent = String(state.accounts.length);
  highRiskCountEl.textContent = String(state.accounts.filter((account) => account.cti >= 55).length);
  communityCountEl.textContent = String(new Set(state.accounts.map((account) => account.community)).size);
  linkCountEl.textContent = String(state.links.length);
}

function updateSelectionChip() {
  const account = getSelectedAccount();
  selectedAccountChip.textContent = account ? `${account.account_id} · CTI ${formatDecimal(account.cti)}` : 'No account selected';
}

function renderAccountSummary() {
  const account = getSelectedAccount();
  if (!account) {
    accountSummaryEl.innerHTML = `
      <div class="detail-card"><div class="label">Status</div><div class="value">Choose a node</div></div>
      <div class="detail-card"><div class="label">CTI</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Community</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Risk band</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Volume</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Velocity</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Alerts</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Dwell time</div><div class="value">--</div></div>
    `;
    return;
  }

  accountSummaryEl.innerHTML = `
    <div class="detail-card"><div class="label">Selected account</div><div class="value">${account.account_id}</div></div>
    <div class="detail-card"><div class="label">CTI</div><div class="value">${formatDecimal(account.cti)}</div></div>
    <div class="detail-card"><div class="label">Community</div><div class="value">${account.community}</div></div>
    <div class="detail-card"><div class="label">Risk band</div><div class="value">${account.riskBand}</div></div>
    <div class="detail-card"><div class="label">Volume</div><div class="value">${account.volume.toLocaleString()}</div></div>
    <div class="detail-card"><div class="label">Velocity</div><div class="value">${account.velocity.toFixed(2)}</div></div>
    <div class="detail-card"><div class="label">Alerts</div><div class="value">${account.alerts}</div></div>
    <div class="detail-card"><div class="label">Dwell time</div><div class="value">${account.dwellTime} days</div></div>
  `;
}

function renderPlainEnglish(statements) {
  if (!statements || statements.length === 0) {
    plainEnglishEl.innerHTML = '<div class="empty-state">SHAP summaries will appear here after the selected account is explained.</div>';
    return;
  }
  plainEnglishEl.innerHTML = statements.map((statement) => `<div class="narrative-item">${statement}</div>`).join('');
}

function localShapFallback(account) {
  const entries = Object.entries(account.featureValues || {}).map(([feature, value]) => {
    const centered = value - 0.5;
    const scale = feature === 'F3924' ? 28 : feature === 'F3836' ? 16 : 12;
    return [feature, Number((centered * scale + (account.cti - 50) / 18).toFixed(4))];
  });
  return Object.fromEntries(entries.sort((left, right) => Math.abs(right[1]) - Math.abs(left[1])));
}

function fetchShap(account) {
  return fetch('/explain/shap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      account_id: account.account_id,
      features: account.featureValues,
      timeseries: account.timeseries,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`SHAP endpoint returned ${response.status}`);
      }
      return response.json();
    })
    .catch((error) => {
      console.warn('using local SHAP fallback', error);
      return {
        shap: localShapFallback(account),
        plain_english: [
          `Synthetic evidence suggests ${account.account_id} is being pulled upward by the strongest local signals.`,
          `Community ${account.community} remains tightly coupled to the selected neighborhood.`,
        ],
      };
    });
}

function buildLocalRingSummary(account) {
  const members = getRingMembers(account);
  const labelRate = members.filter((candidate) => candidate.cti >= 70).length / Math.max(members.length, 1);
  const recentActivityRate = members.filter((candidate) => candidate.cti >= 45).length / Math.max(members.length, 1);
  const stage = labelRate >= 0.45 ? 'Active' : recentActivityRate >= 0.65 ? 'Recruiting' : members.length > 2 ? 'Dispersing' : 'Dormant';
  const stageScore = Number((0.35 + members.length / 10 + labelRate * 0.25).toFixed(3));
  return {
    community_id: account.community,
    members: members.map((candidate) => candidate.account_id),
    stage,
    stage_score: stageScore,
    community_summary: [
      {
        community_id: account.community,
        members: members.length,
        stage,
        stage_score: stageScore,
        label_rate: Number(labelRate.toFixed(3)),
        recent_activity_rate: Number(recentActivityRate.toFixed(3)),
      },
    ],
  };
}

function fetchRingSummary(account) {
  const cacheKey = account.account_id;
  if (state.ringCache.has(cacheKey)) {
    return Promise.resolve(state.ringCache.get(cacheKey));
  }

  return fetch(`/rings/account/${encodeURIComponent(account.account_id)}?out_dir=models/graph`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`ring endpoint returned ${response.status}`);
      }
      return response.json();
    })
    .then((payload) => {
      state.ringCache.set(cacheKey, payload);
      return payload;
    })
    .catch((error) => {
      const fallback = buildLocalRingSummary(account);
      state.ringCache.set(cacheKey, fallback);
      console.warn('using local ring fallback', error);
      return fallback;
    });
}

function renderWaterfall(chartData, account) {
  const width = Math.max(320, document.getElementById('waterfall-shell').clientWidth);
  const height = Math.max(260, document.getElementById('waterfall-shell').clientHeight);
  waterfallSvg.selectAll('*').remove();
  waterfallSvg.attr('viewBox', `0 0 ${width} ${height}`);

  const entries = Object.entries(chartData || {})
    .filter(([, value]) => Number.isFinite(Number(value)))
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]))
    .slice(0, 8)
    .map(([feature, value]) => ({
      feature,
      label: featureDescriptions[feature] || feature,
      value: Number(value),
    }));

  if (entries.length === 0) {
    waterfallSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', '#89a0b3')
      .style('font-size', '13px')
      .text('Select a node to load SHAP contributions');
    return;
  }

  const margin = { top: 12, right: 18, bottom: 62, left: 160 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const rowHeight = innerHeight / entries.length;
  const maxAbs = d3.max(entries, (entry) => Math.abs(entry.value)) || 1;
  const x = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([0, innerWidth]);
  const baseline = x(0);
  const root = waterfallSvg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  root.append('line')
    .attr('x1', baseline)
    .attr('x2', baseline)
    .attr('y1', 0)
    .attr('y2', innerHeight)
    .attr('stroke', 'rgba(255,255,255,0.22)')
    .attr('stroke-dasharray', '4 4');

  root.append('text')
    .attr('x', baseline)
    .attr('y', innerHeight + 30)
    .attr('text-anchor', 'middle')
    .attr('fill', '#89a0b3')
    .style('font-size', '11px')
    .text('Baseline');

  root.selectAll('.waterfall-row')
    .data(entries)
    .enter()
    .append('g')
    .attr('class', 'waterfall-row')
    .attr('transform', (_entry, index) => `translate(0, ${index * rowHeight + 4})`)
    .each(function drawRow(entry) {
      const row = d3.select(this);
      const barHeight = Math.max(12, rowHeight - 12);
      const start = entry.value >= 0 ? baseline : x(entry.value);
      const end = entry.value >= 0 ? x(entry.value) : baseline;
      const barWidth = Math.max(2, Math.abs(end - start));
      const color = entry.value >= 0 ? 'var(--rose)' : 'var(--teal)';

      row.append('text')
        .attr('x', -12)
        .attr('y', barHeight / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', '#eef4fb')
        .style('font-size', '12px')
        .text(entry.feature);

      row.append('text')
        .attr('x', -12)
        .attr('y', barHeight / 2 + 19)
        .attr('text-anchor', 'end')
        .attr('fill', '#89a0b3')
        .style('font-size', '10px')
        .text(entry.label);

      row.append('rect')
        .attr('x', Math.min(start, end))
        .attr('y', 2)
        .attr('width', barWidth)
        .attr('height', barHeight)
        .attr('rx', 8)
        .attr('fill', color)
        .attr('opacity', 0.9)
        .append('title')
        .text(`${entry.feature}: ${entry.value >= 0 ? '+' : ''}${entry.value.toFixed(3)}`);

      row.append('text')
        .attr('x', entry.value >= 0 ? end + 8 : start - 8)
        .attr('y', barHeight / 2 + 4)
        .attr('text-anchor', entry.value >= 0 ? 'start' : 'end')
        .attr('fill', '#eef4fb')
        .style('font-size', '11px')
        .text(`${entry.value >= 0 ? '+' : ''}${entry.value.toFixed(3)}`);
    });

  if (account) {
    const total = entries.reduce((sum, entry) => sum + entry.value, account.cti * 0.04);
    root.append('text')
      .attr('x', innerWidth - 6)
      .attr('y', -2)
      .attr('text-anchor', 'end')
      .attr('fill', '#89a0b3')
      .style('font-size', '11px')
      .text(`Projected contribution ${total.toFixed(2)}`);
  }
}

function renderPlainEnglish(statements) {
  if (!statements || statements.length === 0) {
    plainEnglishEl.innerHTML = '<div class="empty-state">SHAP summaries will appear here after the selected account is explained.</div>';
    return;
  }
  plainEnglishEl.innerHTML = statements.map((statement) => `<div class="narrative-item">${statement}</div>`).join('');
}

function renderRelationshipSummary() {
  const link = getSelectedLink();
  if (link) {
    const source = typeof link.source === 'string' ? link.source : link.source.id;
    const target = typeof link.target === 'string' ? link.target : link.target.id;
    relationshipSummaryEl.innerHTML = `
      <div class="detail-card"><div class="label">Link</div><div class="value">${source} ↔ ${target}</div></div>
      <div class="detail-card"><div class="label">Weight</div><div class="value">${link.weight.toFixed(3)}</div></div>
      <div class="detail-card"><div class="label">Similarity</div><div class="value">${link.similarity.toFixed(3)}</div></div>
      <div class="detail-card"><div class="label">Sync score</div><div class="value">${link.sync_score.toFixed(3)}</div></div>
    `;
    return;
  }

  const account = getSelectedAccount();
  const ringSummary = state.currentRingSummary && state.currentRingSummary.community_summary && state.currentRingSummary.community_summary.length > 0
    ? state.currentRingSummary.community_summary[0]
    : null;

  if (!account || !ringSummary) {
    relationshipSummaryEl.innerHTML = `
      <div class="detail-card"><div class="label">Ring stage</div><div class="value">Select an account</div></div>
      <div class="detail-card"><div class="label">Stage score</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Member count</div><div class="value">--</div></div>
      <div class="detail-card"><div class="label">Action</div><div class="value">--</div></div>
    `;
    return;
  }

  relationshipSummaryEl.innerHTML = `
    <div class="detail-card"><div class="label">Ring stage</div><div class="value">${ringSummary.stage || 'Unknown'}</div></div>
    <div class="detail-card"><div class="label">Stage score</div><div class="value">${Number(ringSummary.stage_score || 0).toFixed(3)}</div></div>
    <div class="detail-card"><div class="label">Member count</div><div class="value">${ringSummary.members || getRingMembers(account).length}</div></div>
    <div class="detail-card"><div class="label">Action</div><div class="value">${account.cti >= 55 ? 'Priority review' : 'Monitor'}</div></div>
  `;
}

function isConnected(accountId) {
  if (!state.hoveredNodeId) {
    return true;
  }
  return state.links.some((link) => {
    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
    const targetId = typeof link.target === 'string' ? link.target : link.target.id;
    return (sourceId === accountId && targetId === state.hoveredNodeId) || (targetId === accountId && sourceId === state.hoveredNodeId) || accountId === state.hoveredNodeId;
  });
}

function updateGraphStyles(linkSelection, nodeSelection, labelSelection) {
  nodeSelection
    .attr('opacity', (node) => {
      if (state.hoveredNodeId && node.account_id !== state.hoveredNodeId && !isConnected(node.account_id)) {
        return 0.25;
      }
      return 1;
    })
    .attr('stroke-width', (node) => {
      if (node.account_id === state.selectedAccountId) return 3.5;
      if (node.account_id === state.hoveredNodeId) return 2.4;
      return 1.4;
    });

  labelSelection.attr('opacity', (node) => {
    if (state.hoveredNodeId && node.account_id !== state.hoveredNodeId && !isConnected(node.account_id)) {
      return 0.25;
    }
    return 1;
  });

  linkSelection
    .attr('opacity', (link) => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      if (state.hoveredNodeId && sourceId !== state.hoveredNodeId && targetId !== state.hoveredNodeId && link.id !== state.hoveredLinkId) {
        return 0.16;
      }
      return link.id === state.selectedLinkId ? 1 : 0.7;
    })
    .attr('stroke', (link) => {
      if (link.id === state.selectedLinkId) return '#ffbf5d';
      if (link.id === state.hoveredLinkId) return '#74a9ff';
      return 'rgba(255,255,255,0.28)';
    });
}

function renderGraph() {
  if (state.simulation) {
    state.simulation.stop();
  }

  const width = Math.max(640, graphShell.clientWidth);
  const height = Math.max(420, graphShell.clientHeight);
  graphSvg.selectAll('*').remove();
  graphSvg.attr('viewBox', `0 0 ${width} ${height}`);

  const zoomLayer = graphSvg.append('g').attr('class', 'zoom-layer');
  const linkLayer = zoomLayer.append('g');
  const nodeLayer = zoomLayer.append('g');
  const labelLayer = zoomLayer.append('g');
  const links = state.links.map((link) => ({ ...link }));
  const nodes = state.accounts.map((account) => ({ ...account }));

  const linkSelection = linkLayer.selectAll('line')
    .data(links, (link) => link.id)
    .join('line')
    .attr('stroke', 'rgba(255,255,255,0.28)')
    .attr('stroke-width', (link) => 1 + link.weight * 3)
    .attr('stroke-linecap', 'round')
    .style('cursor', 'pointer')
    .on('click', (_event, link) => {
      state.selectedLinkId = link.id;
      state.selectedAccountId = typeof link.source === 'string' ? link.source : link.source.id;
      subscribe(state.selectedAccountId);
      renderAll();
      refreshSelectedEvidence();
    })
    .on('mouseenter', (event, link) => {
      state.hoveredLinkId = link.id;
      graphTooltip.style.left = `${event.offsetX}px`;
      graphTooltip.style.top = `${event.offsetY}px`;
      graphTooltip.style.opacity = '1';
      graphTooltip.innerHTML = `
        <strong>${typeof link.source === 'string' ? link.source : link.source.id}</strong> → <strong>${typeof link.target === 'string' ? link.target : link.target.id}</strong><br />
        weight ${link.weight.toFixed(3)} · similarity ${link.similarity.toFixed(3)} · sync ${link.sync_score.toFixed(3)}
      `;
      updateGraphStyles(linkSelection, nodeSelection, labelSelection);
    })
    .on('mouseleave', () => {
      state.hoveredLinkId = null;
      graphTooltip.style.opacity = '0';
      updateGraphStyles(linkSelection, nodeSelection, labelSelection);
    });

  const nodeSelection = nodeLayer.selectAll('circle')
    .data(nodes, (node) => node.account_id)
    .join('circle')
    .attr('r', (node) => 10 + node.cti / 10)
    .attr('fill', (node) => accountRiskColor(node.cti))
    .attr('stroke', 'rgba(255,255,255,0.72)')
    .attr('stroke-width', 1.4)
    .style('cursor', 'pointer')
    .on('click', (_event, node) => selectAccount(node.account_id))
    .on('mouseenter', (event, node) => {
      state.hoveredNodeId = node.account_id;
      graphTooltip.style.left = `${event.offsetX}px`;
      graphTooltip.style.top = `${event.offsetY}px`;
      graphTooltip.style.opacity = '1';
      graphTooltip.innerHTML = `
        <strong>${node.account_id}</strong><br />
        CTI ${node.cti.toFixed(1)} · community ${node.community} · ${node.riskBand}
      `;
      updateGraphStyles(linkSelection, nodeSelection, labelSelection);
    })
    .on('mouseleave', () => {
      state.hoveredNodeId = null;
      graphTooltip.style.opacity = '0';
      updateGraphStyles(linkSelection, nodeSelection, labelSelection);
    })
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended));

  const labelSelection = labelLayer.selectAll('text')
    .data(nodes, (node) => node.account_id)
    .join('text')
    .text((node) => node.account_id)
    .attr('fill', '#eef4fb')
    .attr('font-size', 11)
    .attr('text-anchor', 'middle')
    .attr('pointer-events', 'none')
    .attr('dy', (node) => 22 + node.cti / 14);

  const simulation = d3.forceSimulation(nodes)
    .force('charge', d3.forceManyBody().strength(-280))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('link', d3.forceLink(links).id((node) => node.account_id).distance((link) => 100 - link.weight * 35).strength((link) => 0.4 + link.weight * 0.7))
    .force('collide', d3.forceCollide().radius((node) => 28 + node.cti / 10))
    .on('tick', ticked);

  state.simulation = simulation;

  graphSvg.call(d3.zoom().scaleExtent([0.5, 2.6]).on('zoom', (event) => {
    zoomLayer.attr('transform', event.transform);
  }));

  function ticked() {
    linkSelection
      .attr('x1', (link) => link.source.x)
      .attr('y1', (link) => link.source.y)
      .attr('x2', (link) => link.target.x)
      .attr('y2', (link) => link.target.y);

    nodeSelection
      .attr('cx', (node) => node.x)
      .attr('cy', (node) => node.y);

    labelSelection
      .attr('x', (node) => node.x)
      .attr('y', (node) => node.y);
  }

  function dragstarted(event, node) {
    if (!event.active) {
      simulation.alphaTarget(0.3).restart();
    }
    node.fx = node.x;
    node.fy = node.y;
  }

  function dragged(event, node) {
    node.fx = event.x;
    node.fy = event.y;
  }

  function dragended(event, node) {
    if (!event.active) {
      simulation.alphaTarget(0);
    }
    node.fx = null;
    node.fy = null;
  }

  updateGraphStyles(linkSelection, nodeSelection, labelSelection);
}

function updateModelStatus() {
  if (state.backendHealthy === null) {
    modelStatus.textContent = 'checking';
    backendStatus.textContent = 'checking';
    return;
  }
  modelStatus.textContent = state.backendHealthy ? 'ready' : 'fallback';
  backendStatus.textContent = state.backendHealthy ? 'online' : 'offline';
}

function renderAll() {
  renderAccounts();
  updateSummaryCards();
  updateSelectionChip();
  renderAccountSummary();
  renderRelationshipSummary();
  renderGraph();
  updateModelStatus();
}

function refreshSelectedEvidence() {
  const account = getSelectedAccount();
  if (!account) {
    state.currentRingSummary = null;
    renderPlainEnglish([]);
    renderWaterfall({}, null);
    renderRelationshipSummary();
    return Promise.resolve();
  }

  const requestAccountId = account.account_id;
  renderAccountSummary();

  return Promise.all([
    fetchShap(account),
    fetchRingSummary(account),
  ]).then(([shapPayload, ringPayload]) => {
    if (state.selectedAccountId !== requestAccountId) {
      return;
    }

    state.currentRingSummary = ringPayload;
    renderPlainEnglish(shapPayload.plain_english || []);
    renderWaterfall(shapPayload.shap || {}, account);
    renderRelationshipSummary();
  });
}

function selectAccount(accountId) {
  state.selectedAccountId = accountId;
  state.selectedLinkId = null;
  state.pendingSubscription = accountId;
  subscribe(accountId);
  renderAll();
  return refreshSelectedEvidence();
}

function addGeneratedAccount() {
  const randomId = `acct-${Math.floor(1000 + Math.random() * 9000)}`;
  const account = addAccount(randomId);
  return selectAccount(account.account_id);
}

function probeBackend() {
  return fetch('/health')
    .then((response) => response.json().then((data) => ({ response, data })))
    .then(({ response, data }) => {
      state.backendHealthy = Boolean(response.ok && data.models_loaded);
    })
    .catch(() => {
      state.backendHealthy = false;
    })
    .finally(() => {
      updateModelStatus();
    });
}

function updateAccountFromCti(accountId, cti) {
  const account = state.accounts.find((candidate) => candidate.account_id === accountId);
  if (!account) {
    return addAccount(accountId);
  }
  account.cti = Number(cti);
  account.riskBand = riskBandForCti(account.cti);
  account.featureValues.F3924 = account.cti > 68 ? 1 : 0;
  account.featureValues.F3836 = Number((account.featureValues.F3836 + account.cti / 200).toFixed(3));
  state.shapCache.delete(accountId);
  state.ringCache.delete(accountId);
  rebuildLinks();
  return account;
}

ws.addEventListener('open', () => {
  connectionDot.classList.remove('offline');
  connectionText.textContent = 'Live feed connected';
  if (state.pendingSubscription) {
    subscribe(state.pendingSubscription);
  }
});

ws.addEventListener('close', () => {
  connectionDot.classList.add('offline');
  connectionText.textContent = 'Live feed disconnected';
});

ws.addEventListener('message', (event) => {
  try {
    const message = JSON.parse(event.data);
    if (message.type === 'cti_update') {
      const accountId = message.account_id || state.selectedAccountId;
      updateAccountFromCti(accountId, message.cti);
      renderAll();
      if (state.selectedAccountId === accountId) {
        refreshSelectedEvidence();
      }
    }

    if (message.type === 'brief_ready') {
      const link = document.createElement('a');
      link.href = message.url;
      link.download = message.filename || 'brief.pdf';
      document.body.appendChild(link);
      link.click();
      link.remove();
    }
  } catch (error) {
    console.error(error);
  }
});

simulateBtn.addEventListener('click', addGeneratedAccount);
focusSelectedBtn.addEventListener('click', () => {
  const account = getSelectedAccount();
  if (account) {
    selectAccount(account.account_id);
  }
});
generateSelectedBtn.addEventListener('click', () => {
  const account = getSelectedAccount();
  if (account) {
    generateBrief(account.account_id);
  }
});

window.addEventListener('resize', () => {
  renderGraph();
  const account = getSelectedAccount();
  if (account && state.shapCache.has(account.account_id)) {
    const payload = state.shapCache.get(account.account_id);
    renderWaterfall(payload.shap || {}, account);
  }
});

function init() {
  return loadFeatureNames()
    .then(() => {
      ['acct-1001', 'acct-1002', 'acct-1003', 'acct-2048', 'acct-3072', 'acct-4096'].forEach((accountId) => {
        addAccount(accountId, true);
      });
      state.selectedAccountId = state.accounts[0] ? state.accounts[0].account_id : null;
      state.pendingSubscription = state.selectedAccountId;
      if (state.selectedAccountId) {
        subscribe(state.selectedAccountId);
      }
      renderAll();
      return probeBackend();
    })
    .then(() => refreshSelectedEvidence());
}

init().catch((error) => {
  console.error(error);
  connectionDot.classList.add('offline');
  connectionText.textContent = 'Initialization failed';
});
