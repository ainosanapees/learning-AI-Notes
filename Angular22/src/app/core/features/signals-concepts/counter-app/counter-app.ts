import { Component, inject, signal } from '@angular/core';
import { Counter } from '../../../../services/counter';

@Component({
  selector: 'app-counter-app',
  imports: [],
  templateUrl: './counter-app.html',
  styleUrl: './counter-app.css',
})
export class CounterApp {
  // private readonly count = signal<number>(0);
  //this.count.set(5); --> allowed
  // this.count =signal(7); --> not allowed

  // readonly count = this._count.asReadonly(); -->other code can read it but cannot modify it
  // it protects from accidental modification of the signal from outside the class
    // count = signal<number>(0);
    public counterService = inject(Counter);
    count = this.counterService.count;

  increment()
  {
    this.counterService.increment();
  }

  decrement()
  {
    this.counterService.decrement();
  }

  reset()
  {
    this.counterService.reset();
  }
}
