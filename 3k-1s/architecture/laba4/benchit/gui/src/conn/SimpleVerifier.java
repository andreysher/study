package conn;

import ch.ethz.ssh2.*;

class SimpleVerifier implements ServerHostKeyVerifier {
	private final KnownHosts database;

	/*
	 * This class is being used by to verify keys from e.g. "~/.ssh/known_hosts"
	 */
	public SimpleVerifier(KnownHosts database) {
		if (database == null)
			throw new IllegalArgumentException();

		this.database = database;
	}

	public boolean verifyServerHostKey(String hostname, int port, String serverHostKeyAlgorithm,
			byte[] serverHostKey) throws Exception {
		int result = database.verifyHostkey(hostname, serverHostKeyAlgorithm, serverHostKey);

		switch (result) {
			case KnownHosts.HOSTKEY_IS_OK :

				return true; // We are happy

			case KnownHosts.HOSTKEY_IS_NEW :

				// Unknown host? Blindly accept the key and put it into the cache.
				// Well, you definitely can do better (e.g., ask the user).

				// The following call will ONLY put the key into the memory cache!
				// To save it in a known hosts file, call also "KnownHosts.addHostkeyToFile(...)"
				database.addHostkey(new String[]{hostname}, serverHostKeyAlgorithm, serverHostKey);

				return true;

			case KnownHosts.HOSTKEY_HAS_CHANGED :

				// Close the connection if the hostkey has changed.
				// Better: ask user and add new key to database.
				return false;

			default :
				throw new IllegalStateException();
		}
	}
}