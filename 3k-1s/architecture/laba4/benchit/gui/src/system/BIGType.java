package system;

public enum BIGType {
	/**
	 * The value has the type <code>null</code>.
	 **/
	None,
	/**
	 * The value has the type <code>Integer</code>.
	 **/
	Integer,
	/**
	 * The value has the type <code>Double</code>.
	 **/
	Float,
	/**
	 * The value has the type <code>Boolean</code>.
	 **/
	Boolean,
	/**
	 * The value has the type <code>String</code>. The multipleChoice is a <code>BIGStrings</code>.
	 **/
	Multiple,
	/**
	 * The value has the type <code>String</code>.
	 **/
	String,
	/**
	 * The value has the type <code>BIGStrings</code>. The multipleChoice is a <code>BIGStrings</code>.
	 **/
	List,
	/**
	 * The value has the type <code>BIGStrings</code>. Not yet implemented
	 **/
	// Vector,
	/**
	 * The value has the type <code>String</code>.
	 **/
	FreeText,
	/**
	 * The value has the type <code>String</code>.
	 **/
	RestrictedText;
}