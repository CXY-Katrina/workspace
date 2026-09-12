import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['TEMP'])/'inference-metrics-python'))
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parent
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1.5)
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.goto((root/'index.html').as_uri())
    page.evaluate('document.fonts.ready')
    for w,h in [(1440,900),(1600,1000),(1920,1080),(2048,1320),(390,844)]:
        page.set_viewport_size({'width':w,'height':h})
        measure=page.evaluate('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})')
        assert measure['scrollWidth']<=w,measure
        results.append(measure)
    page.set_viewport_size({'width':1440,'height':900})
    page.locator('[data-select="tpot"]').click()
    assert page.locator('#single-diagram .measure.dim').count()==6
    assert '75 ms/token' in page.locator('#metric-note').inner_text()
    page.locator('[data-select="all"]').click()
    for time,expected in [('0','在途 1 · Decode 0'),('3.5','在途 3 · Decode 2'),('6','在途 2 · Decode 1'),('9','在途 0 · Decode 0')]:
        page.locator('#at-time').fill(time)
        assert expected in page.locator('#concurrency').inner_text()
    page.locator('#at-time').fill('3.5')
    page.locator('#single-diagram svg').screenshot(path=str(root/'single-request.png'))
    page.locator('#system-diagram svg').screenshot(path=str(root/'system-window.png'))
    page.locator('#stats').screenshot(path=str(root/'statistics.png'))
    page.screenshot(path=str(root/'guide-full.png'),full_page=True)
    assert not errors,errors
    browser.close()
(root/'guide-check.json').write_text(json.dumps({'horizontal_containment':results,'interactions':'passed','page_errors':errors,'note':'Explanatory guide intentionally scrolls vertically.'},indent=2),encoding='utf-8')
print('Guide checks passed; screenshots saved.')
