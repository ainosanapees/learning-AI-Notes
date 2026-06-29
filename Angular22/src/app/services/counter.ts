import { computed, Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class Counter {
  private readonly _count = signal<number>(0);  
  
  readonly count =this._count.asReadonly(); //other code can read it but cannot modify it

  // computed signals are derived from other signals and automatically update when the underlying signals change
  readonly doubleCount =computed(() => this.count() * 2);
  readonly tripleCount =computed(() => this.count() * 3);


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
