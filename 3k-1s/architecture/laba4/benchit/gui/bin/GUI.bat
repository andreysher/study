echo ### execute java GUI ###
java -Xmx256m -Xms256m -jar -Dusessl=true -Djavax.net.ssl.trustStore=..\cfg\client.trusts -Djavax.net.ssl.trustStorePassword=BenchIT BenchIT.jar
