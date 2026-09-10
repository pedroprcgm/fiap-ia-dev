import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { AskResponse, Patient } from './models';

// URL da API Node.js (src/api) - ver Secao 7 da especificacao. Configuravel
// via variavel global `window.__API_BASE_URL__` (definida em index.html) para
// facilitar apontar para outro host/porta sem rebuild; padrao localhost:3000.
const DEFAULT_API_BASE_URL = 'http://localhost:3000';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl =
    (globalThis as { __API_BASE_URL__?: string }).__API_BASE_URL__ ?? DEFAULT_API_BASE_URL;

  constructor(private readonly http: HttpClient) {}

  getPatients(): Observable<Patient[]> {
    return this.http.get<Patient[]>(`${this.baseUrl}/patients`);
  }

  ask(patientId: string | null, question: string): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${this.baseUrl}/ask`, {
      patient_id: patientId,
      question,
    });
  }
}
