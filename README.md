Servidor HTTP

Para este segundo laboratorio, se creo un servidor HTTP básico que puede responder a peticiones GET y entregar archivos estáticos al navegador.

## Requisitos previos

Asegurarse de tener instalado Python3 en tu sistema antes de continuar. Puedes comprobarlo ejecutando el siguiente comando:

python3 --version

## Compilación

No es necesario compilar el servidor, ya que está escrito en Python. Solo hay que asegurarse de tener los archivos del servidor en el mismo directorio.

## Ejecución

1. Abre una terminal y navega hasta el directorio donde se encuentran los archivos del servidor.

2. Ejecuta el siguiente comando para iniciar el servidor:

```
python3 serv.py
```

3. Una vez que el servidor esté iniciado, abrir tu navegador web y acceder a la dirección `http://localhost:8080` para interactuar con el servidor.

4. Posteriormente , el servidor responderá a las solicitudes GET y entregará los mensajes en la terminal. Si el archivo solicitado no existe o hay algún otro error, se mostrará un mensaje de error correspondiente.




