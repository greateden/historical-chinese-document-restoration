"""Optional v3 residual cleanup. Only isolated, geometry-supported components.

Decisions use native connected components in wide local strips. Nothing is
removed merely for being dark or inside a column. Connected/short ambiguous
ink is retained byte-for-byte; no character reconstruction is attempted.
"""
import cv2
import numpy as np
from .reconstruction import corridors


def cleanup_residuals(original, baseline, lines, enabled=True):
    out = baseline.copy()
    records = []
    if not enabled:
        return out, records
    sx, sy = lines['scale']
    h, w = baseline.shape
    permitted = list(corridors(lines, baseline.shape))
    for index, line in enumerate(lines['v']):
        # Wide context exposes connections to glyphs on either side.
        x0 = max(0, round((line['x0']-24)*sx))
        x1 = min(w, round((line['x1']+24)*sx))
        y0 = max(0, round((min(q['y0'] for q in lines['v'])-10)*sy))
        y1 = min(h, round((max(q['y1'] for q in lines['v'])+10)*sy))
        a = baseline[y0:y1,x0:x1]
        raw = original[y0:y1,x0:x1]
        edge = max(2,round(4*sx))
        left = np.percentile(a[:,:edge],80,axis=1).astype(np.float32)
        right = np.percentile(a[:,-edge:],80,axis=1).astype(np.float32)
        t = np.linspace(0,1,a.shape[1],dtype=np.float32)
        paper = left[:,None]*(1-t)+right[:,None]*t
        deficit = paper-a
        ink = (deficit>25).astype(np.uint8)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(ink,8)
        local_permitted = np.zeros_like(a)
        for _,(px0,px1,py0,py1) in permitted:
            if px1>x0 and px0<x1 and py1>y0 and py0<y1:
                local_permitted[max(0,py0-y0):min(len(a),py1-y0),max(0,px0-x0):min(a.shape[1],px1-x0)]=1
        for k in range(1,n):
            xx,yy,ww,hh,area = map(int,stats[k])
            cx,cy = centers[k]; gx=cx+x0; gy=cy+y0
            # Only components overlapping the actual detected ruling band.
            if xx+x0 > (line['x1']+2.5)*sx or xx+ww+x0 < (line['x0']-2.5)*sx:
                continue
            if area < max(2,round(.4*sx*sy)):
                continue
            aligned = (xx+x0 >= (line['x0']-8)*sx-1 and
                       xx+ww+x0 <= (line['x1']+8)*sx+1)
            at_frame = any((q['y0']-2)*sy <= gy <= (q['y1']+2)*sy for q in lines['h'])
            dot = at_frame and hh<=20*sy and ww<=20*sx
            vertical = hh>=8*sy and hh>=3*ww
            # Original line tracking: median profile in adjoining 64-work-pixel
            # context must contain sustained ink at this component's centre.
            lo=max(0,yy-round(32*sy)); hi=min(len(a),yy+hh+round(32*sy))
            col=round(cx)
            track=float(np.mean(raw[lo:hi,max(0,col-1):col+2]<180))
            touches = xx==0 or xx+ww==a.shape[1] or yy==0 or yy+hh==len(a)
            pad=max(2,round(2*max(sx,sy)))
            ax=max(0,xx-pad); bx=min(a.shape[1],xx+ww+pad)
            ay=max(0,yy-pad); by=min(len(a),yy+hh+pad)
            component=(labels[ay:by,ax:bx]==k).astype(np.uint8)
            nearby=cv2.dilate(component,np.ones((2*pad+1,2*pad+1),np.uint8))>0
            other=(labels[ay:by,ax:bx]>0)&(labels[ay:by,ax:bx]!=k)
            isolated=not bool(np.any(nearby&other))
            dot_aligned=(xx+x0 >= (line['x0']-8)*sx and xx+ww+x0 <= (line['x1']+8)*sx)
            safe=(dot_aligned if dot else aligned) and not touches and isolated and (dot or (vertical and track>=.45))
            record=dict(kind='A_frame_dot' if dot else 'B_vertical_fragment' if vertical and aligned else 'C_ambiguous',
                        action='remove' if safe else 'preserve',line=index,
                        x0=x0+xx,x1=x0+xx+ww,y0=y0+yy,y1=y0+yy+hh,
                        area=area,track=round(track,4),isolated=isolated)
            records.append(record)
            if not safe:
                continue
            halo=max(1,round(sx))
            mask=cv2.dilate(component,np.ones((2*halo+1,2*halo+1),np.uint8))>0
            mask &= ~other
            mask &= local_permitted[ay:by,ax:bx]>0
            mask &= deficit[ay:by,ax:bx]>1.5
            mask &= np.minimum(left[ay:by],right[ay:by])[:,None]>180
            target=out[y0+ay:y0+by,x0+ax:x0+bx]
            fill=np.rint(paper[ay:by,ax:bx]).clip(0,255).astype(np.uint8)
            target[mask]=fill[mask]
            record['changed_pixels']=int(mask.sum())
            if not record['changed_pixels']:
                record['action']='preserve'
                record['reason']='no_safe_reconstruction_pixels'
    return out, records
