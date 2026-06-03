import os
import json
import hashlib
import zipfile
import logging
from typing import Dict, Any, List, Optional

import pandas as pd

logger = logging.getLogger('muleguard.shieldscan')


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def static_analyze_apk(apk_path: str) -> Dict[str, Any]:
    """Perform static analysis on APK. Uses androguard if available; otherwise fallbacks to basic metadata."""
    meta: Dict[str, Any] = {'apk_path': apk_path, 'sha256': sha256_of_file(apk_path)}
    try:
        from androguard.core.bytecodes.apk import APK
        a = APK(apk_path)
        meta['package_name'] = a.get_package()
        meta['permissions'] = a.get_permissions()
        meta['activities'] = a.get_activities()
        meta['services'] = a.get_services()
        meta['receivers'] = a.get_receivers()
        meta['main_activity'] = a.get_main_activity()
        try:
            certs = a.get_certificates()
            meta['certificates'] = [c.sha256_hexdigest() if hasattr(c, 'sha256_hexdigest') else None for c in certs]
        except Exception:
            meta['certificates'] = []
        meta['note'] = 'Full androguard static analysis'
    except Exception as e:
        logger.warning('Androguard not available or failed: %s', e)
        # fallback minimal analysis: list zip contents
        try:
            with zipfile.ZipFile(apk_path, 'r') as z:
                files = z.namelist()
            meta['contents'] = files[:200]
            meta['note'] = 'Lightweight analysis (androguard missing)'
        except Exception as ee:
            meta['contents'] = []
            meta['note'] = f'Failed lightweight analysis: {ee}'
    return meta


def parse_dynamic_trace(trace_json_path: str) -> Dict[str, Any]:
    if not os.path.exists(trace_json_path):
        raise FileNotFoundError(trace_json_path)
    with open(trace_json_path, 'r') as f:
        data = json.load(f)
    # expect trace to contain keys like 'package', 'device_id', 'network_hosts', 'suspicious_calls'
    return data


def correlate_with_frauddna(apk_meta: Dict[str, Any], frauddna_manifest_path: Optional[str] = None,
                            accounts_events_path: Optional[str] = None) -> Dict[str, Any]:
    """Correlate APK metadata with FraudDNA manifest and optional account-events mapping.

    Returns report with matched patterns and correlated accounts.
    """
    report = {'matches': [], 'accounts': []}
    indicators: List[str] = []
    if apk_meta.get('package_name'):
        indicators.append(str(apk_meta['package_name']))
    indicators.append(apk_meta.get('sha256', ''))

    if frauddna_manifest_path and os.path.exists(frauddna_manifest_path):
        try:
            manifest = pd.read_parquet(frauddna_manifest_path)
            # search textual columns for indicators
            text_cols = [c for c in manifest.columns if manifest[c].dtype == object]
            for ind in indicators:
                if not ind:
                    continue
                hits = pd.DataFrame()
                for col in text_cols:
                    try:
                        mask = manifest[col].astype(str).str.contains(ind, na=False)
                        if mask.any():
                            hits = pd.concat([hits, manifest[mask]])
                    except Exception:
                        continue
                if not hits.empty:
                    # record matched patterns
                    for _, r in hits.drop_duplicates().iterrows():
                        match = {'indicator': ind, 'pattern_id': r.get('pattern_id', None), 'file_path': r.get('file_path', None), 'support_count': r.get('support_count', None)}
                        report['matches'].append(match)
        except Exception as e:
            logger.exception('Failed to read frauddna manifest: %s', e)

    # If accounts_events_path provided, find accounts referencing the APK (e.g., installed_apks or device_apk)
    if accounts_events_path and os.path.exists(accounts_events_path):
        try:
            ae = pd.read_parquet(accounts_events_path)
            # expect 'account_id' and some apk column like 'installed_apks' or 'package_name'
            candidates = []
            if 'installed_apks' in ae.columns:
                for ind in indicators:
                    if not ind:
                        continue
                    mask = ae['installed_apks'].astype(str).str.contains(ind, na=False)
                    if mask.any():
                        candidates.extend(ae[mask]['account_id'].tolist())
            if 'package_name' in ae.columns:
                for ind in indicators:
                    if not ind:
                        continue
                    mask = ae['package_name'].astype(str).str.contains(ind, na=False)
                    if mask.any():
                        candidates.extend(ae[mask]['account_id'].tolist())
            report['accounts'] = sorted(list(set(candidates)))
        except Exception as e:
            logger.exception('Failed to read accounts events: %s', e)

    return report


def generate_apk_correlation_report(apk_path: str, out_dir: str = 'models/shieldscan',
                                    frauddna_manifest_path: Optional[str] = None,
                                    accounts_events_path: Optional[str] = None,
                                    dynamic_trace: Optional[str] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    apk_meta = static_analyze_apk(apk_path)
    dyn = None
    if dynamic_trace:
        try:
            dyn = parse_dynamic_trace(dynamic_trace)
        except Exception as e:
            logger.warning('Failed to parse dynamic trace: %s', e)

    corr = correlate_with_frauddna(apk_meta, frauddna_manifest_path=frauddna_manifest_path, accounts_events_path=accounts_events_path)

    report = {
        'apk_meta': apk_meta,
        'dynamic_trace': dyn,
        'correlation': corr,
        'generated_at': pd.Timestamp.now().isoformat()
    }
    sha = apk_meta.get('sha256', 'unknown')[:8]
    out_path = os.path.join(out_dir, f'report_{os.path.basename(apk_path)}_{sha}.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    return out_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('apk')
    p.add_argument('--frauddna_manifest', default=os.path.join('models', 'frauddna_manifest.parquet'))
    p.add_argument('--accounts_events')
    p.add_argument('--dynamic')
    args = p.parse_args()
    out = generate_apk_correlation_report(args.apk, frauddna_manifest_path=args.frauddna_manifest, accounts_events_path=args.accounts_events, dynamic_trace=args.dynamic)
    print('Report written to', out)
