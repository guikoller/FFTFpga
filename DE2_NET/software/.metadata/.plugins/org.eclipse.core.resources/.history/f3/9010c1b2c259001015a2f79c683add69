/******************************************************************************
* Copyright (c) 2006 Altera Corporation, San Jose, California, USA.           *
* All rights reserved. All use of this software and documentation is          *
* subject to the License Agreement located at the end of this file below.     *
*******************************************************************************                                                                             *
* Date - October 24, 2006                                                     *
* Module - iniche_init.c                                                      *
*                                                                             *                                                                             *
******************************************************************************/

/******************************************************************************
 * NicheStack TCP/IP stack initialization and Operating System Start in main()
 * for Simple Socket Server (SSS) example.
 *
 * This example demonstrates the use of MicroC/OS-II running on NIOS II.
 * In addition it is to serve as a good starting point for designs using
 * MicroC/OS-II and Altera NicheStack TCP/IP Stack - NIOS II Edition.
 *
 * Please refer to the Altera NicheStack Tutorial documentation for details on
 * this software example, as well as details on how to configure the NicheStack
 * TCP/IP networking stack and MicroC/OS-II Real-Time Operating System.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <io.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
/* MicroC/OS-II definitions */
#include "../simple_socket_bsp/HAL/inc/includes.h"

#include "../simple_socket_bsp/system.h"

#include "dm9000a.h"

/* Simple Socket Server definitions */
#include "simple_socket_server.h"
#include "alt_error_handler.h"

/* Nichestack definitions */
#include "../simple_socket_bsp/iniche/src/h/nios2/ipport.h"
#include "../simple_socket_bsp/iniche/src/h/tcpport.h"
#include "../simple_socket_bsp/iniche/src/h/libport.h"
#include "../simple_socket_bsp/iniche/src/nios2/osport.h"
#include "basic_io.h"
#include "LCD.h"
#include "altera_avalon_pio_regs.h"

#define N 1024

/* Definition of task stack for the initial task which will initialize the NicheStack
 * TCP/IP Stack and then initialize the rest of the Simple Socket Server example tasks.
 */
OS_STK    SSSInitialTaskStk[TASK_STACKSIZE];

/* Declarations for creating a task with TK_NEWTASK.
 * All tasks which use NicheStack (those that use sockets) must be created this way.
 * TK_OBJECT macro creates the static task object used by NicheStack during operation.
 * TK_ENTRY macro corresponds to the entry point, or defined function name, of the task.
 * inet_taskinfo is the structure used by TK_NEWTASK to create the task.
 */
TK_OBJECT(to_ssstask);
TK_ENTRY(SSSSimpleSocketServerTask);

struct inet_taskinfo ssstask = {
      &to_ssstask,
      "simple socket server",
      SSSSimpleSocketServerTask,
      4,
      APP_STACK_SIZE,
};

/* SSSInitialTask will initialize the NicheStack
 * TCP/IP Stack and then initialize the rest of the Simple Socket Server example
 * RTOS structures and tasks.
 */

// Funcao para inverter bits (bit reversal)
unsigned int reverse_bits(unsigned int x, int bits) {
    unsigned int y = 0;
    int i = 0;
    for(i = 0; i < bits; i++) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

void SSSInitialTask(void *task_data)
{
  INT8U error_code;
  printf(">>>INIT NOVO 2\n");

  /*
   * Initialize Altera NicheStack TCP/IP Stack - Nios II Edition specific code.
   * NicheStack is initialized from a task, so that RTOS will have started, and
   * I/O drivers are available.  Two tasks are created:
   *    "Inet main"  task with priority 2
   *    "clock tick" task with priority 3
   */
  alt_iniche_init();
  netmain();

  /* Wait for the network stack to be ready before proceeding.
   * iniche_net_ready indicates that TCP/IP stack is ready, and IP address is obtained.
   */
  while (!iniche_net_ready)
    TK_SLEEP(1);

  /* Now that the stack is running, perform the application initialization steps */

  /* Application Specific Task Launching Code Block Begin */

  printf("\nSimple Socket Server starting up\n");

  /* Create the main simple socket server task. */
  //TK_NEWTASK(&ssstask);

  /*create os data structures */
  //SSSCreateOSDataStructs();

  /* create the other tasks */
  //SSSCreateTasks();

  /* Application Specific Task Launching Code Block End */

  /*This task is deleted because there is no need for it to run again */
  //error_code = OSTaskDel(OS_PRIO_SELF);
  //alt_uCOSIIErrorHandler(error_code, 0);
  LCD_Init();
  struct sockaddr_in sa;
  int res;
  int SocketFD;

  int16_t input_signal[N];
  float real[N];
  float imag[N];
  float magnitude[N];
  int i, j, k, n, m;
  int received, sent;

  printf("estou aquii 1\n");
  SocketFD = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
  printf("Socket criado\n");
  memset(&sa, 0, sizeof sa);
  sa.sin_family = AF_INET;
  sa.sin_port = htons(8080); // ALTERAR PORTA A SER UTILIZADA AQUI
  res = inet_pton(AF_INET, "192.168.0.8", &sa.sin_addr); //ALTERAR O IP DO SERVIDOR AQUI
  if (connect(SocketFD, (struct sockaddr *)&sa, sizeof sa) == -1) {
	perror("connection failed");
	close(SocketFD);
	exit(EXIT_FAILURE);
  }

  while (1) {
	  printf("Início do loop\n");

	  int bytes_received = 0;
	  while (bytes_received < sizeof(int16_t) * N) {
		  int r = recv(SocketFD, ((char*)input_signal) + bytes_received, sizeof(int16_t) * N - bytes_received, 0);
		  if (r <= 0) {
			  perror("recv");
			  break;
		  }
		  bytes_received += r;
	  }
	  if (bytes_received != sizeof(int16_t) * N) {
		  printf("Erro ao receber dados: %d bytes\n", bytes_received);
		  break;
	  }

	  printf("Dados Recebidos: %d\n", bytes_received);
	  int idx = 0;
	  for (idx = 0; idx < 4; idx++) {
		  printf("Input %d: %d\n", idx, input_signal[idx]);
	  }
      printf("bit reverse\n");
      // 2. Conversão para float e bit-reversal
      for (i = 0; i < N; i++) {
          unsigned int ri = reverse_bits(i, 10); // log2(1024) = 10
          real[ri] = (float)input_signal[i];
          imag[ri] = 0.0f;
      }
      printf("FFT usando Cooley-Tukey radix-2\n");
      // 3. FFT usando Cooley-Tukey radix-2
      for (n = 2; n <= N; n <<= 1) {
          float angle = -2.0f * M_PI / n;
          float w_real_step = cosf(angle);
          float w_imag_step = sinf(angle);

          for (k = 0; k < N; k += n) {
              float w_real = 1.0f;
              float w_imag = 0.0f;

              for (j = 0; j < n / 2; j++) {
                  int even = k + j;
                  int odd = k + j + n / 2;

                  float t_real = w_real * real[odd] - w_imag * imag[odd];
                  float t_imag = w_real * imag[odd] + w_imag * real[odd];

                  float u_real = real[even];
                  float u_imag = imag[even];

                  real[even] = u_real + t_real;
                  imag[even] = u_imag + t_imag;

                  real[odd] = u_real - t_real;
                  imag[odd] = u_imag - t_imag;

                  // Atualiza o fator W
                  float temp_w_real = w_real;
                  w_real = w_real * w_real_step - w_imag * w_imag_step;
                  w_imag = temp_w_real * w_imag_step + w_imag * w_real_step;
              }
          }
          printf("cabofft\n");
      }
      printf("magnitude\n");
      // 4. Calcular magnitude
      for (i = 0; i < N; i++) {
          magnitude[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
      }

      printf("FFT REALIZADA\n");

      // 5. Enviar os dados de volta (1024 float32 = 4096 bytes)
      sent = send(SocketFD, (char*)magnitude, sizeof(float) * N, 0);
      if (sent != sizeof(float) * N) {
          printf("Erro ao enviar resultado FFT: %d bytes\n", sent);
          break;
      }else{
    	  printf("Dados enviados \n");
      }

      msleep(1);
  }
}

/* Main creates a single task, SSSInitialTask, and starts task scheduler.
 */

int main (int argc, char* argv[], char* envp[])
{
	printf("Inicio main");

  INT8U error_code;

  DM9000A_INSTANCE( DM9000A_0, dm9000a_0 );
  DM9000A_INIT( DM9000A_0, dm9000a_0 );

  /* Clear the RTOS timer */
  OSTimeSet(0);

  /* SSSInitialTask will initialize the NicheStack
   * TCP/IP Stack and then initialize the rest of the Simple Socket Server example
   * RTOS structures and tasks.
   */
  error_code = OSTaskCreateExt(SSSInitialTask,
                             NULL,
                             (void *)&SSSInitialTaskStk[TASK_STACKSIZE],
                             SSS_INITIAL_TASK_PRIORITY,
                             SSS_INITIAL_TASK_PRIORITY,
                             SSSInitialTaskStk,
                             TASK_STACKSIZE,
                             NULL,
                             0);
  alt_uCOSIIErrorHandler(error_code, 0);

  /*
   * As with all MicroC/OS-II designs, once the initial thread(s) and
   * associated RTOS resources are declared, we start the RTOS. That's it!
   */
  OSStart();

  while(1); /* Correct Program Flow never gets here. */
  printf("Fim main");
  return -1;
}

/******************************************************************************
*                                                                             *
* License Agreement                                                           *
*                                                                             *
* Copyright (c) 2006 Altera Corporation, San Jose, California, USA.           *
* All rights reserved.                                                        *
*                                                                             *
* Permission is hereby granted, free of charge, to any person obtaining a     *
* copy of this software and associated documentation files (the "Software"),  *
* to deal in the Software without restriction, including without limitation   *
* the rights to use, copy, modify, merge, publish, distribute, sublicense,    *
* and/or sell copies of the Software, and to permit persons to whom the       *
* Software is furnished to do so, subject to the following conditions:        *
*                                                                             *
* The above copyright notice and this permission notice shall be included in  *
* all copies or substantial portions of the Software.                         *
*                                                                             *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     *
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         *
* DEALINGS IN THE SOFTWARE.                                                   *
*                                                                             *
* This agreement shall be governed in all respects by the laws of the State   *
* of California and by the laws of the United States of America.              *
* Altera does not recommend, suggest or require that this reference design    *
* file be used in conjunction or combination with any other product.          *
******************************************************************************/
