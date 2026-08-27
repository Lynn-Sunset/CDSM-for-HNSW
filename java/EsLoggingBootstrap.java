package phase0;

import org.elasticsearch.logging.Level;
import org.elasticsearch.logging.Logger;
import org.elasticsearch.logging.internal.spi.LoggerFactory;

import java.util.function.Supplier;

/**
 * Injects a java.util.logging-backed ES LoggerFactory before any ES codec
 * class is loaded. ES SPI codec classes (ES813Int8FlatVectorFormat etc.) do
 * a static-init log call; outside the ES bootstrap that provider is null and
 * the ServiceLoader scan crashes. Call {@link #install()} first thing in main.
 */
public final class EsLoggingBootstrap {

  private EsLoggingBootstrap() {
  }

  public static void install() {
    LoggerFactory.setInstance(new LoggerFactory() {
      @Override
      public Logger getLogger(String name) {
        return new JulLogger(name);
      }

      @Override
      public Logger getLogger(Class<?> clazz) {
        return new JulLogger(clazz.getName());
      }
    });
  }

  static final class JulLogger implements Logger {
    private final String name;
    private final java.util.logging.Logger jul;

    JulLogger(String name) {
      this.name = name;
      this.jul = java.util.logging.Logger.getLogger(name);
    }

    private static java.util.logging.Level toJul(Level l) {
      return switch (l) {
        case TRACE, DEBUG -> java.util.logging.Level.FINE;
        case INFO -> java.util.logging.Level.INFO;
        case WARN -> java.util.logging.Level.WARNING;
        case ERROR, FATAL -> java.util.logging.Level.SEVERE;
        case OFF -> java.util.logging.Level.OFF;
        case ALL -> java.util.logging.Level.ALL;
      };
    }

    private static String fmt(String msg, Object... args) {
      if (args == null || args.length == 0) {
        return msg;
      }
      StringBuilder sb = new StringBuilder();
      int a = 0, i = 0;
      while (i < msg.length()) {
        int b = msg.indexOf("{}", i);
        if (b < 0) {
          sb.append(msg, i, msg.length());
          break;
        }
        sb.append(msg, i, b);
        if (a < args.length) {
          sb.append(args[a++]);
        } else {
          sb.append("{}");
        }
        i = b + 2;
      }
      return sb.toString();
    }

    @Override
    public void log(Level level, String message) {
      jul.log(toJul(level), message);
    }

    @Override
    public void log(Level level, Supplier<String> supplier, Throwable throwable) {
      if (jul.isLoggable(toJul(level))) {
        jul.log(toJul(level), supplier.get(), throwable);
      }
    }

    @Override
    public String getName() {
      return name;
    }

    @Override
    public boolean isFatalEnabled() {
      return jul.isLoggable(java.util.logging.Level.SEVERE);
    }

    @Override
    public boolean isErrorEnabled() {
      return jul.isLoggable(java.util.logging.Level.SEVERE);
    }

    @Override
    public boolean isWarnEnabled() {
      return jul.isLoggable(java.util.logging.Level.WARNING);
    }

    @Override
    public boolean isInfoEnabled() {
      return jul.isLoggable(java.util.logging.Level.INFO);
    }

    @Override
    public boolean isDebugEnabled() {
      return jul.isLoggable(java.util.logging.Level.FINE);
    }

    @Override
    public boolean isTraceEnabled() {
      return jul.isLoggable(java.util.logging.Level.FINEST);
    }

    @Override
    public boolean isEnabled(Level level) {
      return jul.isLoggable(toJul(level));
    }

    @Override public void fatal(Supplier<String> s) { if (isFatalEnabled()) log(Level.FATAL, s.get()); }
    @Override public void fatal(Supplier<String> s, Throwable t) { log(Level.FATAL, s, t); }
    @Override public void fatal(String msg) { log(Level.FATAL, msg); }
    @Override public void fatal(String msg, Throwable t) { log(Level.FATAL, msg, t); }
    @Override public void fatal(String msg, Object... args) { if (isFatalEnabled()) log(Level.FATAL, fmt(msg, args)); }
    @Override public void error(Supplier<String> s) { if (isErrorEnabled()) log(Level.ERROR, s.get()); }
    @Override public void error(Supplier<String> s, Throwable t) { log(Level.ERROR, s, t); }
    @Override public void error(String msg) { log(Level.ERROR, msg); }
    @Override public void error(String msg, Throwable t) { log(Level.ERROR, msg, t); }
    @Override public void error(String msg, Object... args) { if (isErrorEnabled()) log(Level.ERROR, fmt(msg, args)); }
    @Override public void warn(Supplier<String> s) { if (isWarnEnabled()) log(Level.WARN, s.get()); }
    @Override public void warn(Supplier<String> s, Throwable t) { log(Level.WARN, s, t); }
    @Override public void warn(String msg) { log(Level.WARN, msg); }
    @Override public void warn(String msg, Throwable t) { log(Level.WARN, msg, t); }
    @Override public void warn(String msg, Object... args) { if (isWarnEnabled()) log(Level.WARN, fmt(msg, args)); }
    @Override public void info(Supplier<String> s) { if (isInfoEnabled()) log(Level.INFO, s.get()); }
    @Override public void info(Supplier<String> s, Throwable t) { log(Level.INFO, s, t); }
    @Override public void info(String msg) { log(Level.INFO, msg); }
    @Override public void info(String msg, Throwable t) { log(Level.INFO, msg, t); }
    @Override public void info(String msg, Object... args) { if (isInfoEnabled()) log(Level.INFO, fmt(msg, args)); }
    @Override public void debug(Supplier<String> s) { if (isDebugEnabled()) log(Level.DEBUG, s.get()); }
    @Override public void debug(Supplier<String> s, Throwable t) { log(Level.DEBUG, s, t); }
    @Override public void debug(String msg) { log(Level.DEBUG, msg); }
    @Override public void debug(String msg, Throwable t) { log(Level.DEBUG, msg, t); }
    @Override public void debug(String msg, Object... args) { if (isDebugEnabled()) log(Level.DEBUG, fmt(msg, args)); }
    @Override public void trace(Supplier<String> s) { if (isTraceEnabled()) log(Level.TRACE, s.get()); }
    @Override public void trace(Supplier<String> s, Throwable t) { log(Level.TRACE, s, t); }
    @Override public void trace(String msg) { log(Level.TRACE, msg); }
    @Override public void trace(String msg, Throwable t) { log(Level.TRACE, msg, t); }
    @Override public void trace(String msg, Object... args) { if (isTraceEnabled()) log(Level.TRACE, fmt(msg, args)); }

    private void log(Level level, String msg, Throwable t) {
      jul.log(toJul(level), msg, t);
    }
  }
}
