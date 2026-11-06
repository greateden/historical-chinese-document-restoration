"""Independent geometry/contrast verifier. PASS is not proof of glyph fidelity."""
import cv2
import numpy as np
from .reconstruction import corridors


def _max_run_1d(mask):
    d=np.diff(np.r_[0,np.asarray(mask,dtype=np.int8),0])
    starts=np.flatnonzero(d==1); ends=np.flatnonzero(d==-1)
    return int((ends-starts).max()) if starts.size else 0


def residual_metrics(gray, lines, work_size=(1342,2048)):
    """All expected corridors, never edit/protect masks or remover statistics.

    Local paper comes from flanks; grey deficit and gap-tolerant continuity
    cover the entire corridor, including off-centre hollow outlines.
    """
    g=cv2.resize(gray,work_size,interpolation=cv2.INTER_AREA)
    normalized=dict(lines,scale=(1,1))
    heat=np.zeros_like(g)
    per=[]
    for orient,(x0,x1,y0,y1) in corridors(normalized,g.shape, padding=6):
        a=g[y0:y1,x0:x1]; a=a if orient=='v' else a.T
        # Exclude crossing frame corners from the line-continuity statistic.
        a=a[20:-20] if len(a)>80 else a
        bg=np.maximum(np.percentile(a[:,:3],80,axis=1),np.percentile(a[:,-3:],80,axis=1))
        deficit=np.maximum(0,bg[:,None]-a)
        prof=np.median(deficit,axis=0)
        occupancy=(deficit>12).mean(axis=0)
        runs=[]
        for col in range(3,a.shape[1]-3):
            m=(deficit[:,col]>12).astype(np.uint8)
            m=cv2.morphologyEx(m[:,None],cv2.MORPH_CLOSE,np.ones((5,1),np.uint8)).ravel()
            runs.append(_max_run_1d(m)/len(m))
        peak=float(prof[3:-3].max()) if a.shape[1]>6 else float(prof.max())
        occ=float(occupancy[3:-3].max()) if a.shape[1]>6 else float(occupancy.max())
        continuity=max(runs,default=0.)
        per.append(dict(orient=orient,band=[x0,x1,y0,y1],median_deficit=round(peak,3),grey_occupancy=round(occ,4),run_fraction=round(continuity,4),**{"pass":peak<=5 and occ<=.2 and continuity<=.1}))
        full=g[y0:y1,x0:x1]; full=full if orient=='v' else full.T
        paper=np.maximum(np.percentile(full[:,:3],80,axis=1),np.percentile(full[:,-3:],80,axis=1))
        z=np.clip((paper[:,None]-full)*4,0,255).astype(np.uint8)
        dst=heat[y0:y1,x0:x1]; np.maximum(dst,z if orient=='v' else z.T,out=dst)
    return per,heat


def native_residual_metrics(gray, lines):
    """Native cross-axis pixels: downsampling must not hide thin outlines."""
    rows = []
    for orient, (x0, x1, y0, y1) in corridors(lines, gray.shape, padding=6):
        a = gray[y0:y1, x0:x1]
        a = a if orient == 'v' else a.T
        cross_scale = lines['scale'][0 if orient == 'v' else 1]
        along_scale = lines['scale'][1 if orient == 'v' else 0]
        edge = max(2, round(3 * cross_scale))
        trim = round(20 * along_scale)
        if len(a) > 4 * trim:
            a = a[trim:-trim]
        paper = np.maximum(np.percentile(a[:, :edge], 80, axis=1),
                           np.percentile(a[:, -edge:], 80, axis=1))
        deficit = np.maximum(0, paper[:, None] - a)
        peak = float(np.median(deficit, axis=0)[edge:-edge].max())
        occupancy = float((deficit > 12).mean(axis=0)[edge:-edge].max())
        rows.append(dict(orient=orient, median_deficit=round(peak, 3),
                         grey_occupancy=round(occupancy, 4),
                         **{'pass': peak <= 5 and occupancy <= .2}))
    return rows


def verify_removal(orig,result,lines_in,edit_mask,protect_mask,stats,work_size=(1342,2048)):
    permitted=np.zeros_like(orig,dtype=np.uint8)
    for _,(x0,x1,y0,y1) in corridors(lines_in,orig.shape): permitted[y0:y1,x0:x1]=1
    outside=undeclared=protected=changed=outside_ink=0
    # No page-sized int16/float delta stacks (original scans are 246 MP).
    for y in range(0,len(orig),256):
        a=orig[y:y+256]; b=result[y:y+256]; d=a!=b
        outside+=int((d & (permitted[y:y+256]==0)).sum())
        undeclared+=int((d & (edit_mask[y:y+256]==0)).sum())
        protected+=int((d & (protect_mask[y:y+256]>0)).sum())
        outside_ink+=int((d & (a<150) & (permitted[y:y+256]==0)).sum())
        changed+=int(d.sum())
    before,_=residual_metrics(orig,lines_in,work_size)
    after,_=residual_metrics(result,lines_in,work_size)
    native = native_residual_metrics(result, lines_in)
    # Independently compare foreground connectivity outside permitted regions.
    a=cv2.resize(orig,work_size,interpolation=cv2.INTER_AREA)<150
    b=cv2.resize(result,work_size,interpolation=cv2.INTER_AREA)<150
    p=cv2.resize(permitted,work_size,interpolation=cv2.INTER_AREA)>0
    a[p]=False; b[p]=False
    ca=cv2.connectedComponents(a.astype(np.uint8),8)[0]-1
    cb=cv2.connectedComponents(b.astype(np.uint8),8)[0]-1
    checks=[dict(check='native_grey_residual',value=native,pass_=bool(native) and all(r['pass'] for r in native)),
            dict(check='outside_geometry_unchanged',value=outside,pass_=outside==0),
            dict(check='outside_mask_unchanged',value=undeclared,pass_=undeclared==0),
            dict(check='protected_core_unchanged',value=protected,pass_=protected==0),
            dict(check='outside_geometry_ink_changes',value=outside_ink,pass_=outside_ink==0),
            dict(check='outside_geometry_connectivity',value=[ca,cb],pass_=np.array_equal(a,b) and ca==cb),
            dict(check='independent_grey_residual',value=after,pass_=all(r['pass'] for r in after) and len(after)>0),
            dict(check='not_a_noop',value=changed,pass_=changed/orig.size>=3e-4),
            dict(check='expected_line_counts',value=[len(lines_in['v']),len(lines_in['h'])],pass_=len(lines_in['v'])==9 and len(lines_in['h'])==2)]
    for c in checks: c['pass']=bool(c.pop('pass_'))
    passed=all(c['pass'] for c in checks)
    return dict(checks=checks,passed=passed,residual_before=before,residual_after=after,native_residual_after=native,
                glyph_evidence='Exact invariance outside corridors and protected pixels; connectivity outside corridors only. No clean glyph ground truth; in-band glyph damage remains unverified.'),passed
