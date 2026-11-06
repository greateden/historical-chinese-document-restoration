"""Conservative, strip-local paper reconstruction using the v2 detector.

No OCR or inferred glyph reconstruction. Ink entering from either flank is
protected across the ruling and reported as ambiguous. Isolated ink wholly
inside a ruling cannot be distinguished from print without a clean reference.
"""
import cv2
import numpy as np


def corridors(lines, shape, padding=8):
    """Geometry-only permitted rectangles, independent of replacement masks."""
    h, w = shape[:2]
    sx, sy = lines['scale']
    for orient in ('v', 'h'):
        for line in lines[orient]:
            if orient == 'v':
                rect = (max(0, int((line['x0']-padding)*sx)),
                        min(w, int((line['x1']+padding)*sx)+1),
                        max(0, int((line['y0']-3)*sy)),
                        min(h, int((line['y1']+3)*sy)+1))
            else:
                rect = (max(0, int((line['x0']-3)*sx)),
                        min(w, int((line['x1']+3)*sx)+1),
                        max(0, int((line['y0']-padding)*sy)),
                        min(h, int((line['y1']+padding)*sy)+1))
            yield orient, rect


def remove_rulings(img, lines):
    """Grayscale uint8, bounded per-strip float workspace; same v2 return API."""
    if img.ndim != 2:
        raise ValueError('v3 requires grayscale input')
    out = img.copy()
    edit = np.zeros_like(img)
    protect = np.zeros_like(img)
    ambiguities, stats = [], []
    for orient, (x0,x1,y0,y1) in corridors(lines, img.shape):
        raw = img[y0:y1,x0:x1]
        a = raw if orient == 'v' else raw.T
        scale = lines['scale'][0 if orient == 'v' else 1]
        along = lines['scale'][1 if orient == 'v' else 0]
        edge = max(2, round(3*scale))
        # Smooth paper estimates along each flank; percentile rejects scattered
        # dark ink without inventing white paper when both flanks are dark.
        left = np.percentile(a[:,:edge], 80, axis=1).astype(np.float32)
        right = np.percentile(a[:,-edge:], 80, axis=1).astype(np.float32)
        k = max(3, round(5*along) | 1)
        left = cv2.GaussianBlur(left[:,None], (1,k), 0).ravel()
        right = cv2.GaussianBlur(right[:,None], (1,k), 0).ravel()
        t = np.linspace(0,1,a.shape[1],dtype=np.float32)
        bg = left[:,None]*(1-t) + right[:,None]*t
        deficit = bg-a
        support = np.zeros(a.shape, np.uint8)
        # Local profile follows slow centre/width changes, includes grey halos.
        block = max(16, round(64*along))
        for start in range(0,len(a),block):
            stop = min(len(a),start+block)
            p = np.median(deficit[start:stop],axis=0)
            mid = len(p)//2
            search = max(2,round(5*scale))
            c = max(edge,mid-search)+int(np.argmax(p[max(edge,mid-search):min(len(p)-edge,mid+search+1)]))
            if p[c] < 6:
                continue
            lo=hi=c
            while lo>edge and p[lo-1]>2: lo-=1
            while hi<len(p)-edge-1 and p[hi+1]>2: hi+=1
            halo=max(1,round(1.5*scale))
            support[start:stop,max(edge,lo-halo):min(len(p)-edge,hi+halo+1)]=1
        # Foreground connected to the flanks: reserve its rows plus AA halo.
        # Never classify short isolated components as safe-to-delete glyphs.
        flank = (deficit>55) & (support==0)
        near = cv2.dilate(support,np.ones((1,max(3,round(5*scale)*2+1)),np.uint8))>0
        cross = (flank & near).any(axis=1).astype(np.uint8)
        cross = cv2.dilate(cross[:,None],np.ones((max(3,round(3*along)*2+1),1),np.uint8)).ravel()>0
        pm = (support>0) & cross[:,None]
        mask = (support>0) & ~pm & (deficit>1.5) & (np.minimum(left,right)[:,None]>180)
        target = out[y0:y1,x0:x1]
        target = target if orient=='v' else target.T
        target[mask] = np.rint(bg[mask]).clip(0,255).astype(np.uint8)
        em = edit[y0:y1,x0:x1]; pr = protect[y0:y1,x0:x1]
        if orient=='h': em=em.T; pr=pr.T
        em[mask]=255; pr[pm]=255
        # Compact local zones, rather than one page-long intersection box.
        n, lab, cc, _ = cv2.connectedComponentsWithStats(pm.astype(np.uint8),8)
        for xx,yy,ww,hh,area in cc[1:]:
            if area < max(2,round(scale*along)): continue
            if orient=='v': box=[x0+int(xx),x0+int(xx+ww),y0+int(yy),y0+int(yy+hh)]
            else: box=[x0+int(yy),x0+int(yy+hh),y0+int(xx),y0+int(xx+ww)]
            ambiguities.append(dict(orient=orient,x0=box[0],x1=box[1],y0=box[2],y1=box[3],n=int(area)))
        stats.append(dict(orient=orient,x0=x0,x1=x1,y0=y0,y1=y1,mask_px=int(mask.sum()),protected_px=int(pm.sum())))
    # A later orthogonal strip must not overwrite an earlier protected zone.
    out[protect>0]=img[protect>0]
    edit[protect>0]=0
    return out, edit, protect, ambiguities, stats
