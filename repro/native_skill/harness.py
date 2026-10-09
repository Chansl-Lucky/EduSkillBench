"""Frozen OpenCode capture adapter, portable root without model-weight dependencies."""
from dataclasses import replace
from pathlib import Path
import json
import subprocess
from .upstream.profiles import EDUCATION_PROMPT
OUT=Path('.')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def require(ok,message):
 if not ok:raise RuntimeError(message)


def gateway():
    return subprocess.check_output(['docker','network','inspect','bridge','--format','{{range .IPAM.Config}}{{.Gateway}}{{end}}'],text=True).strip()

def configure_adapter():
    from benchflow.agents.registry import AGENTS, AGENT_INSTALLERS, _opencode_family_proxy_wrapper_install
    cfg=AGENTS['opencode']
    # The immutable local image already contains these exact runtime versions.
    # Keep the upstream discovery path and provider wrapper unchanged.
    install=('test -x /opt/benchflow/node/bin/node && test -x /opt/benchflow/js-agents/bin/opencode && '
        '/opt/benchflow/node/bin/node --version && /opt/benchflow/js-agents/bin/opencode --version && '+
        _opencode_family_proxy_wrapper_install('opencode','.config/opencode/opencode.json'))
    AGENTS['opencode']=replace(cfg,install_cmd=install)
    AGENT_INSTALLERS['opencode']=install
    from benchflow.acp.container_transport import ContainerTransport
    original=ContainerTransport.receive
    async def receive(self):
        message=await original(self)
        if self._agent_log_path:
            dest=self._agent_log_path.parent.parent/'trajectory/acp_wire.jsonl'
            append(dest,message)
        return message
    ContainerTransport.receive=receive

DB_EXPORT="""import sqlite3,json,pathlib
p=pathlib.Path('/home/agent/.local/share/opencode/opencode.db')
dest=pathlib.Path('/logs/agent/opencode_database.json')
if p.exists():
 con=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
 tables={r[0] for r in con.execute('select name from sqlite_master where type=\"table\"')}
 data={name:[dict(r) for r in con.execute('select * from '+name)] for name in ['session','message','part'] if name in tables}
 dest.write_text(json.dumps(data,ensure_ascii=False));con.close()
"""

def audited_rollout_class():
    from benchflow.rollout import Rollout
    class AuditedRollout(Rollout):
        async def install_agent(self):
            await super().install_agent()
            audit=self._require_rollout_dir()/'capture';audit.mkdir(exist_ok=True)
            probe=await self._env.exec("runuser -u agent -- sh -c 'if [ -d /skills ]; then find -L /home/agent/.config/opencode/skills -name SKILL.md -type f -exec test -r {} \\; -print; else echo NO_SKILL_DIRECTORY; fi'",timeout_sec=15)
            save(audit/'skill_readability.json',dict(return_code=probe.return_code,stdout=probe.stdout,stderr=probe.stderr))
            if self._config.skill_mode=='with-skill':
                require(probe.return_code==0 and 'SKILL.md' in (probe.stdout or ''),'agent cannot read required Skill')
        async def cleanup(self):
            if self._env:
                dest=self._require_rollout_dir()/'capture';dest.mkdir(exist_ok=True)
                # SQLite and logs are collected while the container still exists,
                # including failed and timed-out rollouts. Never include auth files.
                import shlex
                try:
                    result=await self._env.exec('python3 -c '+shlex.quote(DB_EXPORT),timeout_sec=15)
                    save(dest/'db_export_status.json',dict(return_code=result.return_code,stderr=result.stderr))
                    await self._env.download_file('/logs/agent/opencode_database.json',dest/'opencode_database.json')
                except Exception as e:save(dest/'db_export_error.json',dict(error=str(e)[:300]))
                try:await self._env.download_dir('/home/agent/.local/share/opencode/log',dest/'opencode_logs')
                except Exception as e:save(dest/'log_export_error.json',dict(error=str(e)[:300]))
                try:
                    result=await self._env.exec("python3 -c 'import json,pathlib; p=pathlib.Path(\"/home/agent/.config/opencode/opencode.json\"); d=json.loads(p.read_text()); [(v.get(\"options\",{}).pop(\"apiKey\",None)) for v in d.get(\"provider\",{}).values()]; print(json.dumps(d))'",timeout_sec=10)
                    if result.return_code==0:save(dest/'opencode_config.redacted.json',json.loads(result.stdout))
                except Exception as e:save(dest/'config_capture_error.json',dict(error=str(e)[:300]))
            await super().cleanup()
    return AuditedRollout

def captured_evidence(job):
    db=read(job/'capture/opencode_database.json') if (job/'capture/opencode_database.json').exists() else {}
    parts=[];answer_chars=0;tool_count=0;messages={}
    for row in db.get('message',[]):
        data=json.loads(row['data']) if isinstance(row.get('data'),str) else row.get('data',{})
        messages[row['id']]={**data,'session_id':row.get('session_id')}
    for row in db.get('part',[]):
        data=json.loads(row['data']) if isinstance(row.get('data'),str) else row.get('data',{})
        msg=messages.get(row.get('message_id'),{})
        if msg.get('role')!='assistant' or data.get('type') not in ['text','tool']:continue
        if data.get('type')=='text':answer_chars+=len(data.get('text') or '')
        else:tool_count+=1
        parts.append(dict(session_id=msg.get('session_id'),message_id=row.get('message_id'),**data))
    acp=[]
    wire=[]
    for row in rows(job/'trajectory/acp_wire.jsonl'):
        if row.get('method')=='session/update':
            params=row.get('params',{});update=params.get('update',{})
            if update.get('sessionUpdate') in ['agent_message_chunk','tool_call','tool_call_update']:wire.append(params)
    for row in rows(job/'trajectory/acp_trajectory.jsonl'):
        if row.get('type') in ['agent_message','tool_call']:acp.append(row)
    # Database session IDs keep sub-agent content distinct from parent deliverables.
    evidence={'opencode_sessions':db.get('session',[]),'opencode_assistant_parts':parts,'acp_events':acp,'wire_updates':wire}
    text=json.dumps(evidence,ensure_ascii=False,indent=2)
    if len(text)>200000:text=text[:120000]+'\n[... middle omitted ...]\n'+text[-80000:]
    return text,dict(assistant_text_chars=answer_chars,database_tool_parts=tool_count,wire_updates=len(wire),
        has_deliverable=answer_chars>0 or any(p.get('type')=='tool' and p.get('state',{}).get('status')=='completed' for p in parts),
        database_captured=bool(db))
