import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class Counter {
  private readonly _count = signal<number>(0);  
  
  readonly count =this._count.asReadonly(); //other code can read it but cannot modify it

  

  increment()
  {
    this._count.update((value) => value+1);
  }
  decrement()
  {
    this._count.update((value)=> value -1);
  }
  reset()
  {
    this._count.set(0);
  }
}
