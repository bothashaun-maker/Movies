"""Run against an audit ComfyUI server with the H3 extension installed; no render is queued.
Usage: H3_BROWSER_AUDIT_URL=http://127.0.0.1:8199 python tests/browser_h3_autoload.py
Requires Playwright and Chromium. Use a disposable audit server: this saves a test workflow.
"""
import os
from playwright.sync_api import sync_playwright
base=os.environ['H3_BROWSER_AUDIT_URL'].rstrip('/')
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':2200,'height':1100})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('dialog',lambda d: (errors.append(d.message),d.dismiss()))
    page.goto(base+'/?h3_workflow=full-v1',wait_until='networkidle')
    page.wait_for_function('window.app?.graph?._nodes?.length === 17',timeout=60000)
    page.wait_for_timeout(3000)
    assert page.evaluate('app.graph._nodes.length') == 17
    assert 'h3_workflow' not in page.url
    assert not errors,errors
    page.keyboard.press('Escape')
    if os.environ.get('H3_BROWSER_SCREENSHOT'):
        page.screenshot(path=os.environ['H3_BROWSER_SCREENSHOT'])
    template_requests=[]
    page.on('request', lambda request: template_requests.append(request.url) if 'full-workflow.json' in request.url else None)
    page.reload(wait_until='domcontentloaded')
    page.wait_for_function('window.app?.graph?._nodes?.length === 17')
    page.wait_for_timeout(1500)
    assert not template_requests, template_requests
    assert not errors, errors
    print('PASS: direct URL auto-loaded 17 nodes without page errors; refresh did not fetch/reapply the template.')
    browser.close()
