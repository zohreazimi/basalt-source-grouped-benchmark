// Builds a Computers & Geosciences-style manuscript (.docx) from a JSON block list.
const fs=require('fs');
const D=require('docx');
const {Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,WidthType,AlignmentType,ImageRun,BorderStyle,HeadingLevel,PageNumber,Footer,LevelFormat,LineNumberRestartFormat,PageBreak}=D;
const [,, inJson,outDocx,optStr]=process.argv;
const blocks=JSON.parse(fs.readFileSync(inJson,'utf8'));
const opt=Object.assign({double:true,lineNumbers:true,font:'Times New Roman',size:24},optStr?JSON.parse(optStr):{});
const LINE=opt.double?480:300;
function runs(text,base={}){ // inline markup: **bold**, *italic*, _{sub}, ^{sup}
  const out=[]; const re=/(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g; let last=0,m;
  const push=(t,o)=>{ if(t) out.push(new TextRun(Object.assign({text:t,font:opt.font,size:base.size||opt.size},base,o))); };
  while((m=re.exec(text))){ push(text.slice(last,m.index),{}); const tok=m[0];
    if(tok.startsWith('**')) push(tok.slice(2,-2),{bold:true});
    else if(tok.startsWith('*')) push(tok.slice(1,-1),{italics:true});
    else if(tok.startsWith('_{')) push(tok.slice(2,-1),{subScript:true});
    else push(tok.slice(2,-1),{superScript:true});
    last=m.index+tok.length; }
  push(text.slice(last),{}); return out; }
const P=(text,o={})=>new Paragraph(Object.assign({children:runs(text,o.run||{}),spacing:{line:o.line||LINE,after:o.after===undefined?120:o.after,before:o.before||0},alignment:o.align||AlignmentType.LEFT},o.par||{}));
const thin={style:BorderStyle.SINGLE,size:6,color:'000000'}, none={style:BorderStyle.NONE,size:0,color:'FFFFFF'};
function table(b){ const W=9026; const n=b.rows[0].length; let cw=b.widths?b.widths.map(w=>Math.round(w*W)):Array(n).fill(Math.floor(W/n)); cw[n-1]=W-cw.slice(0,n-1).reduce((a,c)=>a+c,0);
  const fs_=b.size||18;
  const rows=b.rows.map((r,i)=>new TableRow({...(i===0?{tableHeader:true}:{}),cantSplit:true,children:r.map((c,j)=>new TableCell({width:{size:cw[j],type:WidthType.DXA},margins:{top:40,bottom:40,left:70,right:70},
    borders:{top:i===0?thin:none,bottom:(i===0||i===b.rows.length-1)?thin:none,left:none,right:none},
    children:[new Paragraph({children:runs(String(c),{size:fs_,bold:i===0}),spacing:{line:240,after:0},alignment:(j===0||b.left)?AlignmentType.LEFT:AlignmentType.CENTER})]}))}));
  return new Table({width:{size:W,type:WidthType.DXA},columnWidths:cw,rows}); }
const children=[];
for(const b of blocks){
  if(b.type==='title') children.push(P(b.text,{run:{bold:true,size:32},line:360,after:240}));
  else if(b.type==='h1') children.push(new Paragraph({heading:HeadingLevel.HEADING_1,children:runs(b.text,{bold:true,size:26}),spacing:{before:360,after:120,line:360},keepNext:true}));
  else if(b.type==='h2') children.push(new Paragraph({heading:HeadingLevel.HEADING_2,children:runs(b.text,{italics:true,size:24}),spacing:{before:240,after:120,line:360},keepNext:true}));
  else if(b.type==='p') children.push(P(b.text,b.opt||{}));
  else if(b.type==='small') children.push(P(b.text,{run:{size:20},line:300}));
  else if(b.type==='bullets') for(const t of b.items) children.push(new Paragraph({numbering:{reference:'bul',level:0},children:runs(t),spacing:{line:LINE,after:60}}));
  else if(b.type==='ref') children.push(new Paragraph({children:runs(b.text,{size:opt.refSize||opt.size}),spacing:{line:LINE,after:60},indent:{left:480,hanging:480}}));
  else if(b.type==='caption') children.push(new Paragraph({children:runs(b.text,{size:22}),spacing:{line:300,after:b.after===undefined?120:b.after,before:b.before||120},keepNext:!!b.keepNext}));
  else if(b.type==='table') { children.push(table(b)); children.push(new Paragraph({children:[],spacing:{after:b.noteAfter||120,line:240}})); }
  else if(b.type==='note') children.push(P(b.text,{run:{size:18},line:260,after:240}));
  else if(b.type==='figure'){ const w=b.width||600; children.push(new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,spacing:{before:240,after:60},children:[new ImageRun({type:'png',data:fs.readFileSync(b.path),transformation:{width:w,height:Math.round(w*b.h/b.w)}})]})); }
  else if(b.type==='pagebreak') children.push(new Paragraph({children:[new PageBreak()]}));
}
const sectionProps={page:{size:{width:11906,height:16838},margin:{top:1440,bottom:1440,left:1440,right:1440}}};
if(opt.lineNumbers) sectionProps.lineNumbers={countBy:1,restart:LineNumberRestartFormat.CONTINUOUS,distance:400};
const doc=new Document({creator:'Authors',title:opt.title||'Manuscript',
  styles:{default:{document:{run:{font:opt.font,size:opt.size}},
    heading1:{run:{font:opt.font,size:26,bold:true,color:'000000'},paragraph:{spacing:{before:360,after:120},keepNext:true}},
    heading2:{run:{font:opt.font,size:24,italics:true,color:'000000'},paragraph:{spacing:{before:240,after:120},keepNext:true}}}},
  numbering:{config:[{reference:'bul',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:560,hanging:280}}}}]}]},
  sections:[{properties:sectionProps,footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:[PageNumber.CURRENT],font:opt.font,size:20})]})]})},children}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(outDocx,b);console.log('wrote',outDocx,b.length);});
