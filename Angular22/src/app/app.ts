import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CounterApp } from './core/features/signals-concepts/counter-app/counter-app';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet,CounterApp],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('Angular22');
}
