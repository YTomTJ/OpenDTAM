#define QUIET_DTAM_PRIMITIVE_CAT(a, ...) a##__VA_ARGS__
#define QUIET_DTAM_IF_WITH_EMPTY_EQUAL_TRUE_INV(p)                                                 \
    QUIET_DTAM_PRIMITIVE_CAT(                                                                      \
        QUIET_DTAM_IIF_, p) // PRODUCES A VALID TOKEN (1) ONLY IF EXPLICITLY DEFINED TO 0,ELSE
                            // PRODUCES AN INVALID TOKEN, WHICH #if treats as 0

#define QUIET_DTAM_IIF_0 1

#ifdef QUIET_DTAM
#if QUIET_DTAM_IF_WITH_EMPTY_EQUAL_TRUE_INV(QUIET_DTAM) == 0
#define cout                                                                                       \
    if (0)                                                                                         \
    cout
#endif
#endif
