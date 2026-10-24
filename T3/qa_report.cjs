// Offline HTML UI checks, executed with a locally installed Playwright package.
const {chromium}=require(process.argv[2]);
const {pathToFileURL}=require('url');
const fs=require('fs'),path=require('path'),crypto=require('crypto');
(async()=>{
  const root=path.resolve(process.argv[3]),out=path.join(root,'browser-qa');fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.EDGE_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});
  let errors=[],network=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
  await page.goto(pathToFileURL(path.join(root,'report.html')).href,{waitUntil:'load'});
  const images=await page.locator('figure img').evaluateAll(xs=>xs.map(x=>x.complete&&x.naturalWidth>0));
  const desktopOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  const buttons=await page.locator('[data-download]').count();
  await page.screenshot({path:path.join(out,'desktop.png')});
  await page.locator('#baseline').screenshot({path:path.join(out,'table1.png')});
  await page.locator('#fig4').screenshot({path:path.join(out,'figure4.png')});
  const [download]=await Promise.all([page.waitForEvent('download'),page.locator('[data-download="table1_reproduced.csv"]').click()]);
  const file=path.join(out,'downloaded_table1.csv');await download.saveAs(file);
  const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const downloadExact=hash(file)===hash(path.join(root,'tables/table1_reproduced.csv'));
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(out,'mobile.png')});
  const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  const result={images:images.length,allImagesLoaded:images.every(Boolean),csvDownloads:buttons,downloadExact,desktopOverflow,mobileOverflow,errors,externalRequests:network};
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
  await browser.close();
  if(images.length!==11||!images.every(Boolean)||buttons<100||!downloadExact||desktopOverflow||mobileOverflow||errors.length||network.length)throw Error('HTML QA failed');
})().catch(e=>{console.error(e);process.exitCode=1});
