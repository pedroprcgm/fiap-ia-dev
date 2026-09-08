import { Component, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService } from './api.service';
import { AskResponse, Patient } from './models';

type Screen = 'select-patient' | 'ask-question' | 'answer';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit {
  protected readonly screen = signal<Screen>('select-patient');

  protected readonly patients = signal<Patient[]>([]);
  protected readonly loadingPatients = signal(false);
  protected readonly patientsError = signal<string | null>(null);

  protected readonly selectedPatient = signal<Patient | null>(null);
  protected readonly question = signal('');

  protected readonly asking = signal(false);
  protected readonly askError = signal<string | null>(null);
  protected readonly answer = signal<AskResponse | null>(null);

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.loadPatients();
  }

  loadPatients(): void {
    this.loadingPatients.set(true);
    this.patientsError.set(null);
    this.api.getPatients().subscribe({
      next: (patients) => {
        this.patients.set(patients);
        this.loadingPatients.set(false);
      },
      error: () => {
        this.patientsError.set(
          'Nao foi possivel carregar a lista de pacientes. Verifique a conexao e tente novamente.'
        );
        this.loadingPatients.set(false);
      },
    });
  }

  selectPatient(patient: Patient): void {
    this.selectedPatient.set(patient);
    this.question.set('');
    this.answer.set(null);
    this.askError.set(null);
    this.screen.set('ask-question');
  }

  backToPatientSelection(): void {
    this.screen.set('select-patient');
  }

  backToQuestion(): void {
    this.screen.set('ask-question');
    this.answer.set(null);
    this.askError.set(null);
  }

  submitQuestion(): void {
    const patient = this.selectedPatient();
    const question = this.question().trim();
    if (!patient || !question) {
      return;
    }

    this.asking.set(true);
    this.askError.set(null);
    this.api.ask(patient.patient_id, question).subscribe({
      next: (response) => {
        this.answer.set(response);
        this.asking.set(false);
        this.screen.set('answer');
      },
      error: () => {
        this.askError.set(
          'Nao foi possivel obter uma resposta agora. Tente novamente em instantes.'
        );
        this.asking.set(false);
      },
    });
  }

  askAnotherQuestion(): void {
    this.question.set('');
    this.answer.set(null);
    this.askError.set(null);
    this.screen.set('ask-question');
  }
}
