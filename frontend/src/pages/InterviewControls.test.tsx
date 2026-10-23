import { afterEach, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import InterviewPage from './InterviewPage';
vi.mock('../lib/api',()=>({widget:{config:async()=>({widget_key:'synthetic',participant_id_mode:'random',audio_available:false,voice_response_available:false,response_mode_default:'text'})}}));
class Socket {
 static OPEN=1;static instance:Socket;readyState=1;
 onopen:(()=>void)|null=null;onmessage:((e:{data:string})=>void)|null=null;
 send=vi.fn();close=vi.fn();constructor(){Socket.instance=this;}
}
let root:Root;
afterEach(async()=>{if(root)await act(async()=>root.unmount());document.body.innerHTML='';vi.unstubAllGlobals();});
async function event(data:object){await act(async()=>Socket.instance.onmessage?.({data:JSON.stringify(data)}));}
async function type(el:HTMLTextAreaElement,value:string){await act(async()=>{Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')!.set!.call(el,value);el.dispatchEvent(new Event('input',{bubbles:true}));});}
it('Enter preserves drafts; explicit Send and addendum controls route answers; breaks can resume',async()=>{
 vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT',true);vi.stubGlobal('WebSocket',Socket);Element.prototype.scrollIntoView=vi.fn();
 const host=document.createElement('div');document.body.append(host);root=createRoot(host);
 await act(async()=>root.render(<MemoryRouter initialEntries={['/interview/synthetic']}><Routes><Route path='/interview/:widgetKey' element={<InterviewPage/>}/></Routes></MemoryRouter>));
 await act(async()=>[...host.querySelectorAll('button')].find(x=>x.textContent?.includes('Start Interview'))!.click());
 await act(async()=>Socket.instance.onopen?.());
 await event({type:'message',role:'agent',text:'What did you use AI for?'});
 const input=host.querySelector('textarea')!;
 await type(input,'An unfinished thought');
 const before=Socket.instance.send.mock.calls.length;
 await act(async()=>input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true})));
 expect(Socket.instance.send.mock.calls.length).toBe(before);
 expect(input.value).toBe('An unfinished thought');
 await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="Send response"]')!.click());
 expect(Socket.instance.send.mock.calls.map(x=>JSON.parse(x[0]))).toContainEqual({type:'message',text:'An unfinished thought'});
 expect(input.disabled).toBe(false);
 await event({type:'message',role:'agent',text:'What would you want next?'});
 await act(async()=>host.querySelector<HTMLInputElement>('input[type=checkbox]')!.click());
 await type(input,'I meant planning a journey.');
 await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="Send response"]')!.click());
 expect(Socket.instance.send.mock.calls.map(x=>JSON.parse(x[0]))).toContainEqual({type:'addendum',text:'I meant planning a journey.'});
 await event({type:'message',role:'agent',text:'Thanks for adding that. What would you want next?'});
 await act(async()=>[...host.querySelectorAll('button')].find(x=>x.textContent==='Take a break')!.click());
 expect(JSON.parse(Socket.instance.send.mock.calls[Socket.instance.send.mock.calls.length-1][0])).toEqual({type:'pause'});
 await event({type:'paused',seconds_remaining:300});
 expect(input.disabled).toBe(true);expect(host.textContent).toContain('Interview paused');
 await act(async()=>[...host.querySelectorAll('button')].find(x=>x.textContent==='Resume interview')!.click());
 expect(JSON.parse(Socket.instance.send.mock.calls[Socket.instance.send.mock.calls.length-1][0])).toEqual({type:'resume'});
 await event({type:'resumed',paused_seconds:1});
 expect(input.disabled).toBe(false);expect(host.textContent).not.toContain('Interview paused');
});
